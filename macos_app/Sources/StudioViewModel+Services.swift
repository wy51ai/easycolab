import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers
import ObjectiveC.runtime

// MARK: - StudioViewModel 扩展：Google OAuth 桥接、自定义 HF 模型热榜与详情、ComfyUI 与网关同步

final class KeepalivePipeLineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()

    func appendAndExtractPayloads(_ chunk: Data) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        pending.append(chunk)
        var payloads: [String] = []
        while let nlIdx = pending.firstIndex(of: 0x0A) {
            let lineData = pending.subdata(in: 0..<nlIdx)
            pending.removeSubrange(0...nlIdx)
            if let lineStr = String(data: lineData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               lineStr.hasPrefix("KEEPALIVE_JSON:") {
                payloads.append(String(lineStr.dropFirst("KEEPALIVE_JSON:".count)))
            }
        }
        return payloads
    }
}

extension StudioViewModel {
    // MARK: - 本地 colab_bridge.py 交互与零配置环境自举
    func ensureBridgePythonBin(installIfMissing: Bool) async -> String? {
        if let cached = cachedBridgePythonBin, FileManager.default.isExecutableFile(atPath: cached) {
            return cached
        }
        let existing = await Task.detached(priority: .userInitiated) {
            LocalRuntimePaths.findReadyBridgePython()
        }.value
        if let existing {
            cachedBridgePythonBin = existing
            return existing
        }
        guard installIfMissing else { return nil }

        guard let sysPy = LocalRuntimePaths.findSystemPython3() else {
            statusMessage = L(
                "未检测到 macOS Python 3 环境，请先安装 Xcode Command Line Tools (xcode-select --install)。",
                "macOS Python 3 not found. Please install Xcode Command Line Tools (xcode-select --install)."
            )
            return nil
        }

        let venvDir = LocalRuntimePaths.managedVenvDir
        let venvPy = venvDir.appendingPathComponent("bin/python").path
        let reqFile = LocalRuntimePaths.resolveBundledOrRepoFile("requirements.txt")
        let hasReqFile = FileManager.default.fileExists(atPath: reqFile)

        statusMessage = L(
            "首次使用正在自动初始化本地 Colab 连接依赖环境 (约 15~30 秒)...",
            "Initializing local Colab bridge Python dependencies for first-time setup (~15–30s)..."
        )
        liveConsoleLines.append(L(
            "[本地初始化] 正在于 ~/Library/Application Support/EasyColab/venv 安装 google-colab-cli 依赖...",
            "[Local Init] Installing google-colab-cli dependencies in ~/Library/Application Support/EasyColab/venv..."
        ))

        let ok = await Task.detached(priority: .userInitiated) { () -> Bool in
            let mkProc = Process()
            mkProc.executableURL = URL(fileURLWithPath: sysPy)
            mkProc.arguments = ["-m", "venv", venvDir.path]
            mkProc.standardOutput = Pipe()
            mkProc.standardError = Pipe()
            do {
                try mkProc.run()
                mkProc.waitUntilExit()
                guard mkProc.terminationStatus == 0 else { return false }
            } catch {
                return false
            }

            let pipProc = Process()
            pipProc.executableURL = URL(fileURLWithPath: venvPy)
            if hasReqFile {
                pipProc.arguments = ["-m", "pip", "install", "-q", "--upgrade", "pip", "-r", reqFile]
            } else {
                pipProc.arguments = [
                    "-m", "pip", "install", "-q", "--upgrade", "pip",
                    "google-colab-cli>=0.7.5",
                    "google-auth>=2.28.0",
                    "google-auth-oauthlib>=1.2.0",
                    "requests>=2.31.0",
                    "websocket-client>=1.7.0"
                ]
            }
            pipProc.standardOutput = Pipe()
            pipProc.standardError = Pipe()
            do {
                try pipProc.run()
                pipProc.waitUntilExit()
                return pipProc.terminationStatus == 0
            } catch {
                return false
            }
        }.value

        if ok, FileManager.default.isExecutableFile(atPath: venvPy) {
            cachedBridgePythonBin = venvPy
            liveConsoleLines.append(L(
                "[本地初始化] Colab 本地连接组件已就绪。",
                "[Local Init] Local Colab bridge environment is ready."
            ))
            return venvPy
        } else {
            statusMessage = L(
                "自动安装本地连接依赖失败，请检查网络或执行 pip install -r requirements.txt。",
                "Failed to install local bridge dependencies. Check your network or run pip install -r requirements.txt."
            )
            return nil
        }
    }

    nonisolated static func extractJSONPayload(from rawOutput: String) -> String {
        let trimmed = rawOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let lines = trimmed.components(separatedBy: .newlines)
        for line in lines.reversed() {
            let candidate = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if candidate.hasPrefix("{") && candidate.hasSuffix("}") {
                return candidate
            }
        }
        return trimmed
    }

    func runBridgeCommand(_ args: [String], installIfMissing: Bool = false) async -> String {
        guard let pyBin = await ensureBridgePythonBin(installIfMissing: installIfMissing) else {
            return ""
        }
        let script = self.bridgeScript
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: pyBin)
                proc.arguments = [script] + args
                let outPipe = Pipe()
                proc.standardOutput = outPipe
                proc.standardError = FileHandle.nullDevice
                do {
                    try proc.run()
                    let data = outPipe.fileHandleForReading.readDataToEndOfFile()
                    proc.waitUntilExit()
                    let out = String(data: data, encoding: .utf8) ?? ""
                    continuation.resume(returning: Self.extractJSONPayload(from: out))
                } catch {
                    continuation.resume(returning: "")
                }
            }
        }
    }

    func refreshOfficialColabAccountUsage() async {
        let out = await runBridgeCommand(["auth_status"], installIfMissing: false)
        guard let data = out.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) else {
            return
        }
        cliLoggedIn = decoded.logged_in ?? false
        if let em = decoded.email, !em.isEmpty {
            accountEmail = em
        } else if !cliLoggedIn {
            accountEmail = ""
        }
        if let bal = decoded.paid_balance {
            officialCUBalance = bal
            totalBudgetCU = bal
        }
        if let rate = decoded.burn_rate_hourly { officialBurnRate = rate }
        if let count = decoded.active_assignments {
            activeVMCount = count
            if count == 0 && !isConnected && (!loadedModelsDetail.isEmpty || vramTotalGB > 0) {
                resetStaleRemoteRuntimeState(reason: L(
                    "云端 Colab 实例当前未运行，已自动清理历史显存状态；点击左侧「唤醒实例并部署」即可启动。",
                    "No active cloud Colab runtime detected; cleared stale VRAM state. Click 'Wake & Deploy' on the left to launch."
                ))
            }
        }
    }

    func startGoogleOAuthLogin(switchingAccount: Bool = false) async {
        let out = await runBridgeCommand(["auth_url"], installIfMissing: true)
        guard let data = out.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let authURLStr = json["auth_url"] as? String,
              let url = URL(string: authURLStr) else {
            if !statusMessage.contains("初始化") && !statusMessage.contains("Initializ") && !statusMessage.contains("Python") {
                statusMessage = L("获取 Google 授权链接失败", "Failed to obtain Google OAuth authorization URL")
            }
            return
        }
        oauthCodeText = ""
        showOAuthCodeInput = true
        NSWorkspace.shared.open(url)
        statusMessage = switchingAccount
            ? L("已在浏览器打开 Google 账号选择页，请选择新账号授权并粘贴授权码（点取消可保留原账号）。", "Opened Google account chooser in browser. Authorize a new account and paste the code (Cancel keeps current account).")
            : L("已在浏览器打开 Google 授权页，请允许访问后粘贴授权码。", "Opened Google authorization page in browser. Grant access and paste the auth code.")
    }

    func logoutGoogleAccount() async {
        stopKeepaliveDaemon()
        _ = await runBridgeCommand(["auth_logout"], installIfMissing: false)
        cliLoggedIn = false
        accountEmail = ""
        showOAuthCodeInput = false
        oauthCodeText = ""
        officialCUBalance = nil
        officialBurnRate = nil
        activeVMCount = 0
        colabServerURL = ""
        resetStaleRemoteRuntimeState(reason: L(
            "已退出当前 Google 账号绑定；可点击左侧「绑定」登录其他账号。",
            "Signed out of Google account. Click 'Bind' on the left to sign in with another account."
        ))
    }

    func submitGoogleOAuthCode() async {
        let code = oauthCodeText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !code.isEmpty else { return }
        let prevEmail = accountEmail
        let wasLoggedIn = cliLoggedIn
        statusMessage = L("正在验证授权码并同步账户余额...", "Verifying authorization code & syncing CU balance...")
        let out = await runBridgeCommand(["auth_code", code], installIfMissing: true)
        guard let data = out.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) else {
            statusMessage = L("授权验证失败，请重试。", "Authorization verification failed. Please try again.")
            return
        }
        if decoded.logged_in == true {
            let newEmail = decoded.email ?? ""
            if wasLoggedIn && (prevEmail != newEmail || !colabServerURL.isEmpty) {
                colabServerURL = ""
                resetStaleRemoteRuntimeState(reason: "")
            }
            cliLoggedIn = true
            accountEmail = newEmail
            showOAuthCodeInput = false
            oauthCodeText = ""
            officialCUBalance = decoded.paid_balance
            officialBurnRate = decoded.burn_rate_hourly
            activeVMCount = decoded.active_assignments ?? 0
            if let bal = decoded.paid_balance {
                totalBudgetCU = bal
            }
            let emailTag = newEmail.isEmpty ? "" : " (\(displayAccountLabel))"
            statusMessage = L(
                "Google 账号绑定成功\(emailTag)！可用额度: \(String(format: "%.2f", decoded.paid_balance ?? 0)) CU",
                "Google account bound\(emailTag)! Available balance: \(String(format: "%.2f", decoded.paid_balance ?? 0)) CU"
            )
        } else {
            statusMessage = L(
                "授权码无效: \(decoded.error ?? "请重新获取")",
                "Invalid authorization code: \(decoded.error ?? "please generate a new code")"
            )
        }
    }

    func wakeColabHeadless() async {
        guard cliLoggedIn else {
            statusMessage = L("请先绑定 Google 账号后再使用免网页一键唤醒。", "Please bind your Google account before using headless 1-click wake.")
            return
        }
        stopKeepaliveDaemon()
        isWakingColab = true
        statusMessage = L(
            "正在通过 Google Colab API 后台分配 \(wakeGPUChoice) 实例并部署多模型引擎...",
            "Provisioning \(wakeGPUChoice) runtime via Google Colab API & deploying multi-model engine..."
        )
        liveConsoleLines = [L(
            "[Colab-CLI] 正在启动免网页一键唤醒流程 (\(wakeGPUChoice) GPU)...",
            "[Colab-CLI] Starting headless 1-click wake (\(wakeGPUChoice) GPU)..."
        )]

        guard let pyBin = await ensureBridgePythonBin(installIfMissing: true) else {
            isWakingColab = false
            return
        }
        let script = self.bridgeScript
        let gpu = self.wakeGPUChoice
        let password = UUID().uuidString + UUID().uuidString
        let idleMin = self.idleTimeoutMinutes
        let appPy = self.appPyPath

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: pyBin)
                proc.environment = ProcessInfo.processInfo.environment.merging([
                    "EASYCOLAB_ADMIN_PASSWORD": password,
                    "EASYCOLAB_IDLE_TIMEOUT_MINUTES": "\(idleMin)"
                ]) { _, new in new }
                proc.arguments = [script, "wake", "--gpu", gpu, "--app-py", appPy]
                let pipe = Pipe()
                proc.standardOutput = pipe
                proc.standardError = pipe

                pipe.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
                    let lines = chunk.components(separatedBy: .newlines)
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
                    Task { @MainActor in
                        for line in lines {
                            if line.hasPrefix("WAKE_URL:") {
                                let url = String(line.dropFirst("WAKE_URL:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                                self.colabServerURL = url
                                self.adminPassword = password
                                self.liveConsoleLines.append(L(
                                    "成功获取云端隧道地址: \(self.maskedURLDisplay(url))",
                                    "Obtained public tunnel URL: \(self.maskedURLDisplay(url))"
                                ))
                                self.statusMessage = L(
                                    "云端 \(gpu) 实例已唤醒就绪，正在同步模型状态...",
                                    "Cloud \(gpu) instance is ready. Syncing model status..."
                                )
                            } else if line.hasPrefix("WAKE_ERROR:") {
                                self.liveConsoleLines.append(line)
                                self.statusMessage = L("唤醒失败: \(line)", "Wake failed: \(line)")
                            } else {
                                let clean = line.replacingOccurrences(of: "WAKE_LOG: ", with: "")
                                self.liveConsoleLines.append(clean)
                                self.statusMessage = clean
                            }
                        }
                    }
                }

                do {
                    try proc.run()
                    proc.waitUntilExit()
                } catch {
                    Task { @MainActor in
                        self.statusMessage = L(
                            "启动唤醒进程失败: \(error.localizedDescription)",
                            "Failed to launch wake process: \(error.localizedDescription)"
                        )
                    }
                }
                pipe.fileHandleForReading.readabilityHandler = nil
                continuation.resume()
            }
        }

        isWakingColab = false
        await refreshOfficialColabAccountUsage()
        if !colabServerURL.isEmpty {
            await checkColabStatus()
            await ensureKeepaliveDaemonRunning()
        }
    }

    func stopKeepaliveDaemon() {
        keepaliveDaemonStdoutPipe?.fileHandleForReading.readabilityHandler = nil
        try? keepaliveDaemonStdinPipe?.fileHandleForWriting.close()
        if let proc = keepaliveDaemonProcess, proc.isRunning {
            proc.terminate()
        }
        keepaliveDaemonProcess = nil
        keepaliveDaemonStdinPipe = nil
        keepaliveDaemonStdoutPipe = nil
        keepaliveProxyWSConnected = false
    }

    func handleKeepaliveDaemonPayloadLine(_ jsonLine: String) {
        guard let data = jsonLine.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) else {
            return
        }
        if let logged = decoded.logged_in {
            cliLoggedIn = logged
        }
        if let wsConn = decoded.ws_connected {
            keepaliveProxyWSConnected = wsConn
        }
        if let bal = decoded.paid_balance {
            officialCUBalance = bal
            totalBudgetCU = bal
        }
        if let rate = decoded.burn_rate_hourly {
            officialBurnRate = rate
        }
        if let count = decoded.active_assignments {
            activeVMCount = count
        }
    }

    func ensureKeepaliveDaemonRunning() async {
        guard cliLoggedIn, isConnected, !isWakingColab else { return }
        if let existing = keepaliveDaemonProcess, existing.isRunning {
            return
        }
        guard !isStartingKeepaliveDaemon else { return }
        isStartingKeepaliveDaemon = true
        defer { isStartingKeepaliveDaemon = false }

        guard let pyBin = await ensureBridgePythonBin(installIfMissing: false) else { return }
        guard cliLoggedIn, isConnected, !isWakingColab else { return }
        if let existing = keepaliveDaemonProcess, existing.isRunning {
            return
        }

        stopKeepaliveDaemon()

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pyBin)
        proc.arguments = [self.bridgeScript, "keepalive_daemon"]
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        proc.standardInput = stdinPipe
        proc.standardOutput = stdoutPipe
        proc.standardError = FileHandle.nullDevice

        let lineBuffer = KeepalivePipeLineBuffer()

        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else { return }
            let completeLines = lineBuffer.appendAndExtractPayloads(chunk)

            if !completeLines.isEmpty {
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    for payload in completeLines {
                        self.handleKeepaliveDaemonPayloadLine(payload)
                    }
                }
            }
        }

        proc.terminationHandler = { [weak self] exitedProc in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.keepaliveDaemonProcess === exitedProc {
                    self.keepaliveDaemonStdoutPipe?.fileHandleForReading.readabilityHandler = nil
                    try? self.keepaliveDaemonStdinPipe?.fileHandleForWriting.close()
                    self.keepaliveDaemonProcess = nil
                    self.keepaliveDaemonStdinPipe = nil
                    self.keepaliveDaemonStdoutPipe = nil
                    self.keepaliveProxyWSConnected = false
                }
            }
        }

        do {
            try proc.run()
            self.keepaliveDaemonProcess = proc
            self.keepaliveDaemonStdinPipe = stdinPipe
            self.keepaliveDaemonStdoutPipe = stdoutPipe
        } catch {
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            try? stdinPipe.fileHandleForWriting.close()
        }
    }

    func keepaliveColabRuntime() async {
        guard cliLoggedIn, !isWakingColab else { return }
        if isConnected {
            await ensureKeepaliveDaemonRunning()
            if keepaliveDaemonProcess?.isRunning == true {
                return
            }
        }
        let out = await runBridgeCommand(["keepalive"])
        guard let data = out.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) else {
            return
        }
        if let bal = decoded.paid_balance {
            officialCUBalance = bal
            totalBudgetCU = bal
        }
        if let rate = decoded.burn_rate_hourly { officialBurnRate = rate }
        if let count = decoded.active_assignments { activeVMCount = count }
    }

    func startBackgroundTelemetryLoop() {
        heartbeatTask?.cancel()
        heartbeatTask = Task {
            if !normalizedBaseURL.isEmpty {
                await checkColabStatus(silent: true)
            }
            var tick = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_500_000_000)
                tick += 1
                if isConnected {
                    // 生成任务进行中由 pollColabJob 附带 telemetry 实时更新图表，仅每 15s 低频同步一次全局状态，避免争抢 Gradio 隧道
                    if !isGenerating || tick % 6 == 0 {
                        await checkColabStatus(silent: true)
                    }
                    if cliLoggedIn && !isWakingColab {
                        await ensureKeepaliveDaemonRunning()
                    }
                } else {
                    if keepaliveDaemonProcess != nil {
                        stopKeepaliveDaemon()
                    }
                    if !normalizedBaseURL.isEmpty && tick % 4 == 0 {
                        await checkColabStatus(silent: true)
                    }
                }

                // 未连接云端实例时每 ~40s 低频刷新一次官方 CU 余额与活跃分配数；已连接时由常驻 keepalive_daemon 实时回传
                if tick % 16 == 0 && cliLoggedIn && !isWakingColab && !isConnected {
                    await refreshOfficialColabAccountUsage()
                } else if tick % 24 == 0 && cliLoggedIn && !isWakingColab && isConnected && keepaliveDaemonProcess?.isRunning != true {
                    await keepaliveColabRuntime()
                }
            }
        }
    }

    // MARK: - 本地历史媒体库（支持 MP4 视频与 PNG/JPG 图片，兼容旧目录）
    func refreshLocalMedia() {
        var allURLs: [URL] = []
        for dir in [localSaveDir] + legacySaveDirs {
            if let urls = try? FileManager.default.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.creationDateKey],
                options: [.skipsHiddenFiles]
            ) {
                allURLs.append(contentsOf: urls)
            }
        }

        let validExts: Set<String> = ["mp4", "png", "jpg", "jpeg", "webp"]
        var seenNames = Set<String>()
        let filtered = allURLs.filter { url in
            guard validExts.contains(url.pathExtension.lowercased()) else { return false }
            return seenNames.insert(url.lastPathComponent).inserted
        }
        let items: [LocalMediaItem] = filtered.compactMap { url in
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
            let date = (attrs?[.creationDate] as? Date) ?? Date.distantPast
            let ext = url.pathExtension.lowercased()
            let fname = url.lastPathComponent
            return LocalMediaItem(
                url: url,
                filename: fname,
                isImage: ext != "mp4",
                createdAt: date,
                elapsedSeconds: mediaElapsedMap[fname]
            )
        }.sorted(by: { $0.createdAt > $1.createdAt })

        self.localMediaItems = items
        if currentMediaURL == nil, let first = items.first {
            selectMedia(first.url)
        }
    }

    func selectMedia(_ url: URL) {
        currentMediaURL = url
        let ext = url.pathExtension.lowercased()
        if ext == "mp4" {
            currentPreviewImage = nil
            player = AVPlayer(url: url)
            player?.play()
        } else {
            player?.pause()
            player = nil
            currentPreviewImage = NSImage(contentsOf: url)
        }
    }

    func applyStatusResponse(_ decoded: ColabStatusResponse) {
        isConnected = true
        gpuName = decoded.gpu_name ?? "Colab GPU"
        vramFreeGB = decoded.vram_free_gb ?? 0.0
        vramTotalGB = decoded.vram_total_gb ?? 0.0
        vramUsedGB = decoded.vram_used_gb ?? 0.0
        ramTotalGB = decoded.ram_total_gb ?? ramTotalGB
        ramUsedGB = decoded.ram_used_gb ?? ramUsedGB
        ramFreeGB = decoded.ram_free_gb ?? ramFreeGB
        diskTotalGB = decoded.disk_total_gb ?? diskTotalGB
        diskUsedGB = decoded.disk_used_gb ?? diskUsedGB
        diskFreeGB = decoded.disk_free_gb ?? diskFreeGB
        shmTotalGB = decoded.shm_total_gb ?? shmTotalGB
        shmUsedGB = decoded.shm_used_gb ?? shmUsedGB
        shmFreeGB = decoded.shm_free_gb ?? shmFreeGB
        driveTotalGB = decoded.drive_total_gb ?? driveTotalGB
        driveUsedGB = decoded.drive_used_gb ?? driveUsedGB
        driveFreeGB = decoded.drive_free_gb ?? driveFreeGB
        recordTelemetrySample()
        if let remoteCustom = decoded.custom_models {
            mergeRemoteCustomModels(remoteCustom)
        }
        loadedModelId = decoded.loaded_model_id
        if let ids = decoded.loaded_model_ids {
            loadedModelIds = ids
        } else if let single = decoded.loaded_model_id, decoded.model_loaded == true {
            loadedModelIds = [single]
        } else if decoded.any_model_loaded == false {
            loadedModelIds = []
        }
        if let details = decoded.loaded_models_detail {
            loadedModelsDetail = details.map { item in
                var clean = item
                clean.name = stripEmoji(item.name)
                if let m = item.mode { clean.mode = stripEmoji(m) }
                return clean
            }
        } else if decoded.any_model_loaded == false {
            loadedModelsDetail = []
        }
        modelLoaded = loadedModelIds.contains(selectedModelId) || (decoded.model_loaded ?? false)
        currentModelName = decoded.current_model ?? L(Self.defaultModelNameZh, Self.defaultModelNameEn)
        loadedContextLengthTokens = decoded.loaded_context_length ?? 0

        if let remoteCtx = decoded.context_length {
            if remoteCtx == contextLengthTokens {
                remoteContextSynced = true
            } else if !remoteContextSynced && Date().timeIntervalSince(lastContextSyncAttemptAt) >= 8.0 {
                lastContextSyncAttemptAt = Date()
                Task { await syncContextLengthToColab() }
            }
        }

        if let files = decoded.files, !files.isEmpty {
            modelFiles = files
        }
        downloadedCount = decoded.downloaded_count ?? downloadedCount
        totalFileCount = decoded.total_count ?? activeModelSpec.defaultFiles.count
        allDownloaded = decoded.all_downloaded ?? (downloadedCount == totalFileCount && totalFileCount > 0)
        totalDiskSizeGB = decoded.total_size_gb ?? totalDiskSizeGB
        drivePersisted = decoded.drive_persisted ?? false
        bf16ShardsCount = decoded.bf16_shards_count ?? 0
        isDownloadingWeights = decoded.is_downloading ?? false
        isLoadingModel = decoded.is_loading_vram ?? false
        syncMacOSRuntimeActivityAssertion()
        if cliLoggedIn && !isWakingColab && keepaliveDaemonProcess?.isRunning != true {
            Task { await ensureKeepaliveDaemonRunning() }
        }
        if let p = decoded.task_progress { modelTaskProgress = p }

        uptimeSeconds = decoded.uptime_seconds ?? uptimeSeconds
        cuBurnRatePerHour = decoded.cu_burn_rate_per_hour ?? cuBurnRatePerHour
        sessionCUUsed = decoded.session_cu_used ?? sessionCUUsed
        idleRemainingSeconds = decoded.idle_remaining_seconds ?? idleRemainingSeconds
        if let remoteIdle = decoded.idle_timeout_minutes {
            if remoteIdle == idleTimeoutMinutes {
                remoteIdleSynced = true
            } else if Date().timeIntervalSince(lastIdleSyncAttemptAt) >= 8.0 {
                remoteIdleSynced = false
                lastIdleSyncAttemptAt = Date()
                let targetIdle = idleTimeoutMinutes
                Task { await updateIdleTimeoutConfig(targetIdle) }
            }
        }
        apiRequestCount = decoded.api_request_count ?? apiRequestCount
        if decoded.api_key_configured == false && !remoteApiKeySynced {
            remoteApiKeySynced = true
            Task { await syncApiKeyToColab(initializeOnly: true) }
        }
        if let lines = decoded.console_lines, !lines.isEmpty {
            liveConsoleLines = lines
        }
        if let cf = decoded.comfyui {
            comfyRunning = cf.is_running ?? false
            comfyStarting = cf.is_starting ?? false
            comfySyncingDrive = cf.is_syncing_drive ?? false
            comfyDownloadingWeight = cf.is_downloading_weight ?? false
            comfyInstallingPackId = cf.installing_pack_id ?? ""
            comfyProgress = cf.progress ?? 0.0
            if let sm = cf.status_message, !sm.isEmpty {
                comfyStatusMessage = sm
            }
            comfyPublicURL = cf.public_url ?? ""
            comfyQueueRunning = cf.queue_running ?? 0
            comfyQueuePending = cf.queue_pending ?? 0
            if let pl = cf.persistence_label, !pl.isEmpty {
                comfyPersistenceLabel = pl
            }
            comfyDrivePersisted = cf.drive_persisted ?? drivePersisted
            if let pr = cf.persist_root, !pr.isEmpty {
                comfyPersistRoot = pr
            }
            comfyCustomNodesCount = cf.custom_nodes_count ?? 0
            comfyModelsCount = cf.models_count ?? 0
            comfyWorkflowsCount = cf.workflows_count ?? 0
            if let outs = cf.recent_outputs {
                comfyRecentOutputs = outs
            }
            if let packs = cf.preset_packs, !packs.isEmpty {
                comfyPresetPacks = packs
            }
        }
    }

    func openComfyUIInBrowser() {
        guard let url = URL(string: comfyPublicURL), !comfyPublicURL.isEmpty else { return }
        NSWorkspace.shared.open(url)
    }

    // MARK: - 自定义 HuggingFace 模型仓库注册与管理

    func loadPersistedCustomModels() {
        guard let data = UserDefaults.standard.data(forKey: customModelsDefaultsKey),
              let dtos = try? JSONDecoder().decode([CustomHFModelDTO].self, from: data) else { return }
        customModels = dtos.map { OpenModelSpec.fromCustomDTO($0) }
    }

    func savePersistedCustomModels() {
        let dtos: [CustomHFModelDTO] = customModels.map { spec in
            let modStr: String
            switch spec.modality {
            case .videoAudio, .video: modStr = "video_audio"
            case .image: modStr = "image"
            case .llmVLM, .comfyUI: modStr = "llm_vlm"
            }
            return CustomHFModelDTO(
                id: spec.id,
                name: spec.nameZh,
                repo_id: spec.repoId,
                include_pattern: spec.includePattern,
                modality: modStr,
                pipeline_arch: nil,
                category: spec.summaryZh,
                approx_size_gb: spec.approxSizeGB,
                is_custom: true
            )
        }
        if let data = try? JSONEncoder().encode(dtos) {
            UserDefaults.standard.set(data, forKey: customModelsDefaultsKey)
        }
    }

    func mergeRemoteCustomModels(_ remoteDTOs: [CustomHFModelDTO]) {
        var existingById: [String: OpenModelSpec] = [:]
        for item in customModels { existingById[item.id] = item }
        for dto in remoteDTOs {
            existingById[dto.id] = OpenModelSpec.fromCustomDTO(dto)
        }
        let orderedIds = remoteDTOs.map { $0.id } + customModels.map { $0.id }.filter { id in !remoteDTOs.contains(where: { $0.id == id }) }
        customModels = orderedIds.compactMap { existingById[$0] }
        savePersistedCustomModels()
    }

    func normalizeHFRepoId(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["https://huggingface.co/", "http://huggingface.co/", "https://hf.co/", "http://hf.co/", "huggingface.co/", "hf.co/"] {
            if s.lowercased().hasPrefix(prefix) {
                s = String(s.dropFirst(prefix.count))
                break
            }
        }
        let parts = s.split(separator: "/").map(String.init)
        if parts.count >= 2 {
            return "\(parts[0])/\(parts[1])"
        }
        return s
    }

    func fetchTrendingHFModels(category: String? = nil, searchQuery: String = "", forceRefresh: Bool = false) async {
        let cat = (category ?? trendingHFCategory).trimmingCharacters(in: .whitespacesAndNewlines)
        let sq = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        if trendingHFCategory != cat {
            trendingHFCategory = cat
        }
        isLoadingTrendingHF = true
        defer { isLoadingTrendingHF = false }

        let base = normalizedBaseURL
        if isConnected, !base.isEmpty {
            var comps = URLComponents(string: "\(base)/minimax_api/hf_trending")
            comps?.queryItems = [
                URLQueryItem(name: "category", value: cat),
                URLQueryItem(name: "limit", value: "12"),
                URLQueryItem(name: "hf_token", value: hfToken),
                URLQueryItem(name: "force_refresh", value: forceRefresh ? "true" : "false"),
                URLQueryItem(name: "search_query", value: sq)
            ]
            if let url = comps?.url, var req = try? adminRequest(url) {
                req.timeoutInterval = 12
                if let (data, resp) = try? await adminSession.data(for: req),
                   (resp as? HTTPURLResponse)?.statusCode == 200,
                   let dec = try? JSONDecoder().decode(HFTrendingAPIResponse.self, from: data),
                   let models = dec.models, !models.isEmpty {
                    trendingHFModels = models
                    return
                }
            }
        }

        let expandQS = "&expand[]=safetensors&expand[]=gguf&expand[]=likes&expand[]=downloads&expand[]=pipeline_tag&expand[]=tags&expand[]=gated"
        let encodedSQ = sq.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        let searchQS = encodedSQ.isEmpty ? "" : "&search=\(encodedSQ)"

        let queryPaths: [String]
        switch cat {
        case "gguf":
            queryPaths = ["/api/models?sort=trendingScore&direction=-1&filter=gguf&limit=20\(searchQS)\(expandQS)"]
        case "llm":
            if sq.isEmpty {
                queryPaths = [
                    "/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-generation&limit=18\(expandQS)",
                    "/api/models?sort=trendingScore&direction=-1&pipeline_tag=image-text-to-text&limit=10\(expandQS)"
                ]
            } else {
                queryPaths = ["/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-generation&limit=20\(searchQS)\(expandQS)"]
            }
        case "image_video":
            queryPaths = [
                "/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-to-image&limit=12\(searchQS)\(expandQS)",
                "/api/models?sort=trendingScore&direction=-1&pipeline_tag=image-to-video&limit=10\(searchQS)\(expandQS)"
            ]
        default:
            queryPaths = ["/api/models?sort=trendingScore&direction=-1&limit=22\(searchQS)\(expandQS)"]
        }

        let tok = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        var parsed: [HFTrendingModelItem] = []
        var seen = Set<String>()

        for qpath in queryPaths {
            var rawArr: [[String: Any]]? = nil
            for host in ["https://huggingface.co", "https://hf-mirror.com"] {
                guard let url = URL(string: "\(host)\(qpath)") else { continue }
                var req = URLRequest(url: url)
                req.timeoutInterval = 10
                req.setValue("EasyColab-macOS/2.0", forHTTPHeaderField: "User-Agent")
                if !tok.isEmpty {
                    req.setValue("Bearer \(tok)", forHTTPHeaderField: "Authorization")
                }
                if let (data, resp) = try? await URLSession.shared.data(for: req),
                   (resp as? HTTPURLResponse)?.statusCode == 200,
                   let arr = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] {
                    rawArr = arr
                    break
                }
            }
            guard let arr = rawArr else { continue }
            for obj in arr {
                let rid = ((obj["id"] as? String) ?? (obj["modelId"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                guard rid.contains("/"), !seen.contains(rid) else { continue }
                seen.insert(rid)
                let ridLow = rid.lowercased()
                let ptag = ((obj["pipeline_tag"] as? String) ?? "").lowercased()
                let tags = ((obj["tags"] as? [String]) ?? []).map { $0.lowercased() }
                let isGGUF = ridLow.contains("gguf") || tags.contains("gguf")
                let gatedVal = obj["gated"]
                let isGated: Bool = {
                    if let b = gatedVal as? Bool { return b }
                    if let s = gatedVal as? String {
                        let sl = s.lowercased()
                        return !sl.isEmpty && sl != "false" && sl != "0" && sl != "none"
                    }
                    return false
                }()
                if sq.isEmpty && cat == "all" && !isGGUF {
                    if UNSUPPORTED_HF_PIPELINE_TAGS.contains(ptag) { continue }
                    if !ptag.isEmpty && !SUPPORTED_TRENDING_PIPELINE_TAGS.contains(ptag) { continue }
                    if ptag.isEmpty && !tags.contains("transformers") && !tags.contains("diffusers") && !tags.contains("text-generation") {
                        continue
                    }
                }
                let modality: String
                if ptag.contains("video") || tags.contains("text-to-video") || tags.contains("image-to-video") {
                    modality = "video_audio"
                } else if ptag.contains("text-to-image") || ptag.contains("image-to-image") || (tags.contains("diffusers") && !isGGUF) {
                    modality = "image"
                } else {
                    modality = "llm_vlm"
                }
                let shortName = rid.split(separator: "/").last.map(String.init) ?? rid
                let sizing = analyzeHFModelSizing(obj, repoId: rid, isGGUF: isGGUF)
                parsed.append(HFTrendingModelItem(
                    repo_id: rid,
                    name: shortName,
                    pipeline_tag: ptag.isEmpty ? (isGGUF ? "gguf" : "text-generation") : ptag,
                    modality: modality,
                    is_gguf: isGGUF,
                    is_gated: isGated,
                    suggested_pattern: isGGUF ? "*Q4_K_M*.gguf" : "",
                    likes: (obj["likes"] as? Int) ?? 0,
                    downloads: (obj["downloads"] as? Int) ?? 0,
                    param_count: sizing.paramCount,
                    param_label: sizing.paramLabel,
                    est_vram_gb: sizing.estVramGb,
                    deploy_status: sizing.deployStatus,
                    deploy_label_zh: sizing.deployZh,
                    deploy_label_en: sizing.deployEn
                ))
                if parsed.count >= 12 { break }
            }
            if parsed.count >= 12 { break }
        }
        if !parsed.isEmpty || !sq.isEmpty {
            trendingHFModels = parsed
        }
    }

    func fetchHFModelDetail(repoId: String) async {
        let cleanId = normalizeHFRepoId(repoId)
        guard cleanId.contains("/") else { return }
        showHFModelDetailOverlay = true
        isLoadingHFModelDetail = true
        defer { isLoadingHFModelDetail = false }

        let base = normalizedBaseURL
        let tok = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        if isConnected, !base.isEmpty {
            var comps = URLComponents(string: "\(base)/minimax_api/hf_model_detail")
            comps?.queryItems = [
                URLQueryItem(name: "repo_id", value: cleanId),
                URLQueryItem(name: "hf_token", value: tok)
            ]
            if let url = comps?.url, var req = try? adminRequest(url) {
                req.timeoutInterval = 14
                if let (data, resp) = try? await adminSession.data(for: req),
                   (resp as? HTTPURLResponse)?.statusCode == 200,
                   var dec = try? JSONDecoder().decode(HFModelDetailInfo.self, from: data),
                   dec.ok == true {
                    dec.readme_excerpt = cleanHFReadmeExcerpt(dec.readme_excerpt)
                    activeHFModelDetail = dec
                    return
                }
            }
        }

        // Offline / direct HuggingFace API fallback
        var metaObj: [String: Any]? = nil
        for host in ["https://huggingface.co", "https://hf-mirror.com"] {
            guard let url = URL(string: "\(host)/api/models/\(cleanId)?blobs=true") else { continue }
            var req = URLRequest(url: url)
            req.timeoutInterval = 12
            req.setValue("EasyColab-macOS/2.0", forHTTPHeaderField: "User-Agent")
            if !tok.isEmpty {
                req.setValue("Bearer \(tok)", forHTTPHeaderField: "Authorization")
            }
            if let (data, resp) = try? await URLSession.shared.data(for: req),
               (resp as? HTTPURLResponse)?.statusCode == 200,
               let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                metaObj = dict
                break
            }
        }
        guard let meta = metaObj else { return }

        let rid = ((meta["id"] as? String) ?? (meta["modelId"] as? String) ?? cleanId).trimmingCharacters(in: .whitespacesAndNewlines)
        let ridLow = rid.lowercased()
        let ptag = ((meta["pipeline_tag"] as? String) ?? "").lowercased()
        let tags = ((meta["tags"] as? [String]) ?? []).map { $0.lowercased() }
        let ggufDict = meta["gguf"] as? [String: Any]
        let isGGUF = ridLow.contains("gguf") || tags.contains("gguf") || (ggufDict != nil && !(ggufDict?.isEmpty ?? true))
        let gatedVal = meta["gated"]
        let isGated: Bool = {
            if let b = gatedVal as? Bool { return b }
            if let s = gatedVal as? String {
                let sl = s.lowercased()
                return !sl.isEmpty && sl != "false" && sl != "0" && sl != "none"
            }
            return false
        }()
        let modality: String
        if ptag.contains("video") || tags.contains("text-to-video") || tags.contains("image-to-video") {
            modality = "video_audio"
        } else if ptag.contains("text-to-image") || ptag.contains("image-to-image") || (tags.contains("diffusers") && !isGGUF) {
            modality = "image"
        } else {
            modality = "llm_vlm"
        }
        let cardData = meta["cardData"] as? [String: Any]
        let configData = meta["config"] as? [String: Any]

        var lic = (cardData?["license"] as? String) ?? ""
        if lic.isEmpty, let licArr = cardData?["license"] as? [String] {
            lic = licArr.joined(separator: ", ")
        }
        if lic.isEmpty { lic = L("未标注 (见模型页)", "Unspecified") }

        var baseModel = (cardData?["base_model"] as? String) ?? ""
        if baseModel.isEmpty, let bmArr = cardData?["base_model"] as? [String] {
            baseModel = bmArr.prefix(2).joined(separator: ", ")
        }

        var archStr = ""
        if let archs = configData?["architectures"] as? [String], let first = archs.first {
            archStr = first
        } else if let gArch = ggufDict?["architecture"] as? String {
            archStr = gArch
        } else if let mType = configData?["model_type"] as? String {
            archStr = mType
        }
        let ctxLen = (ggufDict?["context_length"] as? NSNumber)?.intValue ?? 0
        let langs: [String] = {
            if let arr = cardData?["language"] as? [String] { return Array(arr.prefix(8)) }
            if let s = cardData?["language"] as? String { return [s] }
            return []
        }()
        let usedStorageBytes = (meta["usedStorage"] as? NSNumber)?.doubleValue ?? 0.0
        let usedStorageGb = usedStorageBytes > 0 ? ((usedStorageBytes / 1_073_741_824.0) * 100).rounded() / 100 : 0.0

        var weightFiles: [HFModelWeightFileItem] = []
        var totalWeightFilesGb: Double = 0.0
        if let sibs = meta["siblings"] as? [[String: Any]] {
            for sib in sibs {
                let fname = ((sib["rfilename"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                let fl = fname.lowercased()
                guard fl.hasSuffix(".gguf") || fl.hasSuffix(".safetensors") || fl.hasSuffix(".bin") || fl.hasSuffix(".pth") else { continue }
                let lfs = sib["lfs"] as? [String: Any]
                let fBytes = (sib["size"] as? NSNumber)?.doubleValue ?? (lfs?["size"] as? NSNumber)?.doubleValue ?? 0.0
                let sizeGb = fBytes > 0 ? ((fBytes / 1_073_741_824.0) * 100).rounded() / 100 : 0.0
                totalWeightFilesGb += sizeGb
                let vramNeed = sizeGb > 0 ? ((sizeGb * 1.12 + 1.0) * 10).rounded() / 10 : 0.0
                let fStatus: String
                let fZh: String
                let fEn: String
                if vramNeed <= 0 {
                    fStatus = "unknown"; fZh = "体积待测"; fEn = "Size TBD"
                } else if vramNeed <= 14.5 {
                    fStatus = "great"; fZh = "T4(15G)可跑"; fEn = "T4 15G Ready"
                } else if vramNeed <= 22.5 {
                    fStatus = "good"; fZh = "L4(24G)可跑"; fEn = "L4 24G Ready"
                } else if vramNeed <= 39.0 {
                    fStatus = "a100"; fZh = "需A100(40G)"; fEn = "A100 40G"
                } else if vramNeed <= 78.0 {
                    fStatus = "heavy"; fZh = "需A100(80G)"; fEn = "A100 80G"
                } else {
                    fStatus = "too_large"; fZh = "超单卡显存"; fEn = ">80G VRAM"
                }
                weightFiles.append(HFModelWeightFileItem(
                    filename: fname,
                    size_gb: sizeGb,
                    fit_status: fStatus,
                    fit_zh: fZh,
                    fit_en: fEn
                ))
                if weightFiles.count >= 36 { break }
            }
        }

        let fallbackGb = (!isGGUF && totalWeightFilesGb > 0) ? totalWeightFilesGb : usedStorageGb
        let sizing = analyzeHFModelSizing(meta, repoId: rid, isGGUF: isGGUF, fallbackStorageGb: fallbackGb)

        var readmeExcerpt = ""
        for host in ["https://huggingface.co", "https://hf-mirror.com"] {
            guard let rURL = URL(string: "\(host)/\(rid)/raw/main/README.md") else { continue }
            var rReq = URLRequest(url: rURL)
            rReq.timeoutInterval = 6
            rReq.setValue("EasyColab-macOS/2.0", forHTTPHeaderField: "User-Agent")
            if !tok.isEmpty {
                rReq.setValue("Bearer \(tok)", forHTTPHeaderField: "Authorization")
            }
            if let (rData, rResp) = try? await URLSession.shared.data(for: rReq),
               (rResp as? HTTPURLResponse)?.statusCode == 200,
               let rawMd = String(data: rData.prefix(32768), encoding: .utf8) {
                readmeExcerpt = cleanHFReadmeExcerpt(rawMd)
                if !readmeExcerpt.isEmpty { break }
            }
        }

        let shortName = rid.split(separator: "/").last.map(String.init) ?? rid
        let ownerName = (meta["author"] as? String) ?? (rid.split(separator: "/").first.map(String.init) ?? "")
        let lastMod = String(((meta["lastModified"] as? String) ?? "").prefix(10))

        activeHFModelDetail = HFModelDetailInfo(
            ok: true,
            repo_id: rid,
            name: shortName,
            author: ownerName,
            pipeline_tag: ptag.isEmpty ? (isGGUF ? "gguf" : "text-generation") : ptag,
            modality: modality,
            is_gguf: isGGUF,
            is_gated: isGated,
            license: lic,
            base_model: baseModel,
            architecture: archStr,
            context_length: ctxLen,
            languages: langs,
            likes: (meta["likes"] as? Int) ?? 0,
            downloads: (meta["downloads"] as? Int) ?? 0,
            last_modified: lastMod,
            used_storage_gb: usedStorageGb,
            param_count: sizing.paramCount,
            param_label: sizing.paramLabel,
            est_vram_gb: sizing.estVramGb,
            deploy_status: sizing.deployStatus,
            deploy_label_zh: sizing.deployZh,
            deploy_label_en: sizing.deployEn,
            weight_files: weightFiles,
            readme_excerpt: readmeExcerpt
        )
    }

    func registerCustomHFModel(autoDownload: Bool = false) async {
        let repo = normalizeHFRepoId(customRepoInput)
        guard repo.contains("/"), repo.split(separator: "/").count == 2 else {
            customModelErrorMessage = L(
                "请输入合法的 HuggingFace 仓库地址（格式：作者/仓库名，如 unsloth/Qwen3.8-27B-GGUF）",
                "Please enter a valid HuggingFace Repo ID (format: owner/repo, e.g. unsloth/Qwen3.8-27B-GGUF)"
            )
            return
        }
        customModelErrorMessage = ""
        isRegisteringCustomModel = true
        defer { isRegisteringCustomModel = false }

        let inc = customIncludePatternInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let customName = customDisplayNameInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let modPref = customModalityInput

        let base = normalizedBaseURL
        if isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/custom_model/register") {
            do {
                var req = try adminRequest(url)
                req.httpMethod = "POST"
                req.timeoutInterval = 35
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                req.httpBody = try JSONSerialization.data(withJSONObject: [
                    "repo_id": repo,
                    "include_pattern": inc,
                    "custom_name": customName,
                    "display_name": customName,
                    "modality": modPref,
                    "modality_hint": modPref,
                    "hf_token": hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
                ])
                let (data, _) = try await adminSession.data(for: req)
                let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
                if decoded.action_ok == false {
                    let srvMsg = decoded.action_message ?? ""
                    if !srvMsg.contains("all_rfilenames") {
                        customModelErrorMessage = srvMsg.isEmpty ? L("无法解析该 HuggingFace 仓库", "Failed to resolve HuggingFace repository") : srvMsg
                        return
                    }
                } else {
                    applyStatusResponse(decoded)
                    if let reg = decoded.registered_model {
                        let spec = OpenModelSpec.fromCustomDTO(reg)
                        if let idx = customModels.firstIndex(where: { $0.id == spec.id }) {
                            customModels[idx] = spec
                        } else {
                            customModels.append(spec)
                        }
                        savePersistedCustomModels()
                        switchSelectedModel(spec)
                    }
                    if let am = decoded.action_message, !am.isEmpty {
                        statusMessage = am
                    }
                    showAddCustomModelSheet = false
                    customRepoInput = ""
                    customIncludePatternInput = ""
                    customDisplayNameInput = ""
                    if autoDownload {
                        await startRemoteDownloadWeights()
                    }
                    return
                }
            } catch {
                // 若远端旧版或网络波动，回退为本地先注册
            }
        }

        let slugBase = (repo + (inc.isEmpty ? "" : "_\(inc)"))
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9]+", with: "_", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        let modelId = "custom_" + String(slugBase.prefix(40))
        let resolvedMod: String
        if modPref == "auto" {
            let lower = (repo + " " + inc).lowercased()
            if lower.contains("video") || lower.contains("wan") || lower.contains("ltx") || lower.contains("minimax") || lower.contains("hunyuan") {
                resolvedMod = "video_audio"
            } else if lower.contains("flux") || lower.contains("image") || lower.contains("sdxl") || lower.contains("z-image") {
                resolvedMod = "image"
            } else {
                resolvedMod = "llm_vlm"
            }
        } else {
            resolvedMod = modPref
        }
        let repoShort = String(repo.split(separator: "/").last ?? Substring(repo))
        let finalName = customName.isEmpty ? (inc.isEmpty ? repoShort : "\(repoShort) (\(inc))") : customName
        let archGuess = (repo.lowercased().contains("gguf") || inc.lowercased().contains(".gguf")) ? "gguf" : (resolvedMod == "image" ? "diffusers_image" : "transformers")
        let dto = CustomHFModelDTO(
            id: modelId,
            name: finalName,
            repo_id: repo,
            include_pattern: inc,
            modality: resolvedMod,
            pipeline_arch: archGuess,
            category: "自定义 HuggingFace 仓库 (\(repo))",
            approx_size_gb: 0.0,
            is_custom: true
        )
        let spec = OpenModelSpec.fromCustomDTO(dto)
        if let idx = customModels.firstIndex(where: { $0.id == spec.id }) {
            customModels[idx] = spec
        } else {
            customModels.append(spec)
        }
        savePersistedCustomModels()
        switchSelectedModel(spec)
        showAddCustomModelSheet = false
        customRepoInput = ""
        customIncludePatternInput = ""
        customDisplayNameInput = ""
        statusMessage = L(
            "已添加自定义模型「\(finalName)」(\(repo))，连接 Colab 后即可并发拉取至 Google Drive。",
            "Added custom model '\(finalName)' (\(repo)). Connect to Colab to download weights to Google Drive."
        )
    }

    func removeCustomHFModel(_ spec: OpenModelSpec) async {
        guard spec.isCustom else { return }
        customModels.removeAll { $0.id == spec.id }
        savePersistedCustomModels()
        if selectedModelId == spec.id {
            switchSelectedModel(OPEN_MODEL_CATALOG[0])
        }
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/custom_model/remove") else { return }
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 20
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "model_id": spec.id,
                "delete_weights": false
            ])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
        } catch {}
    }

    func startRemoteComfyUI(forceRestart: Bool = false, autoOpenBrowser: Bool = true) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/start") else {
            statusMessage = L(
                "请先唤醒或连接云端 G4 实例后再启动 ComfyUI。",
                "Please wake or connect to a cloud G4 instance before starting ComfyUI."
            )
            return
        }
        comfyStarting = true
        comfyProgress = 0.04
        let tip = forceRestart
            ? L("正在重启云端 ComfyUI 并重新加载自定义节点...", "Restarting cloud ComfyUI & reloading custom nodes...")
            : L("正在部署云端 ComfyUI 并挂载 Google Drive 持久化软链...", "Deploying cloud ComfyUI & mounting Google Drive persistence symlinks...")
        comfyStatusMessage = tip
        statusMessage = tip

        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 30
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "unload_existing": comfyUnloadOthersOnStart,
                "force_restart": forceRestart
            ])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)

            // 持续轮询直到启动完成并拿到 trycloudflare 公网画布地址
            for _ in 0..<180 {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await checkColabStatus(silent: true)
                statusMessage = comfyStatusMessage
                if comfyRunning && !comfyPublicURL.isEmpty && !comfyStarting {
                    statusMessage = L(
                        "ComfyUI 全屏节点工作台已就绪！正在为您在浏览器中打开画布...",
                        "ComfyUI fullscreen node canvas is ready! Opening in your browser..."
                    )
                    if autoOpenBrowser {
                        openComfyUIInBrowser()
                    }
                    return
                }
                if !comfyStarting && !comfyRunning {
                    break
                }
            }
        } catch {
            comfyStarting = false
            statusMessage = L(
                "启动 ComfyUI 请求失败: \(error.localizedDescription)",
                "Failed to start ComfyUI: \(error.localizedDescription)"
            )
        }
    }

    func stopRemoteComfyUI() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/stop") else { return }
        statusMessage = L(
            "正在同步自定义节点与工作流到 Google Drive 并停止 ComfyUI...",
            "Syncing custom nodes & workflows to Google Drive and stopping ComfyUI..."
        )
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 35
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
            statusMessage = comfyStatusMessage
        } catch {
            statusMessage = L(
                "停止 ComfyUI 失败: \(error.localizedDescription)",
                "Failed to stop ComfyUI: \(error.localizedDescription)"
            )
        }
    }

    func freeRemoteComfyUIVRAM() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/free_vram") else { return }
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 20
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
            if let am = decoded.action_message, !am.isEmpty {
                statusMessage = am
            }
        } catch {}
    }

    func syncRemoteComfyUIToDrive() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/sync_drive") else { return }
        comfySyncingDrive = true
        statusMessage = L(
            "正在将 ComfyUI 自定义节点、模型与工作流同步至 Google Drive...",
            "Syncing ComfyUI custom nodes, models & workflows to Google Drive..."
        )
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 90
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
            if let am = decoded.action_message, !am.isEmpty {
                statusMessage = am
            }
        } catch {
            comfySyncingDrive = false
            statusMessage = L(
                "同步至 Google Drive 失败: \(error.localizedDescription)",
                "Failed to sync to Google Drive: \(error.localizedDescription)"
            )
        }
    }

    func startComfyWeightDownload() async {
        let src = comfyWeightSourceInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !src.isEmpty else {
            statusMessage = L(
                "请先粘贴 HuggingFace 模型直链、下载 URL 或 仓库 ID",
                "Please paste a Hugging Face direct link, download URL, or repo ID first"
            )
            return
        }
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/download_weight") else {
            statusMessage = L("请先连接 Colab 实例后再下载权重", "Please connect to a Colab instance before downloading weights")
            return
        }
        comfyDownloadingWeight = true
        comfyProgress = 0.05
        statusMessage = L(
            "正在通过云端 8 线程通道下载权重至 ComfyUI/models/\(comfyWeightSubfolder)...",
            "Downloading weights via 8-thread cloud channel to ComfyUI/models/\(comfyWeightSubfolder)..."
        )
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 25
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "url_or_repo": src,
                "filename": comfyWeightFilenameInput.trimmingCharacters(in: .whitespacesAndNewlines),
                "subfolder": comfyWeightSubfolder,
                "hf_token": hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
            ])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)

            while comfyDownloadingWeight {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await checkColabStatus(silent: true)
                statusMessage = comfyStatusMessage
            }
        } catch {
            comfyDownloadingWeight = false
            statusMessage = L(
                "提交权重下载失败: \(error.localizedDescription)",
                "Failed to start weight download: \(error.localizedDescription)"
            )
        }
    }

    func installComfyPresetPack(_ pack: ComfyPresetPackItem, openWhenReady: Bool = true) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/install_pack") else {
            statusMessage = L(
                "请先在左侧唤醒或连接云端 G4 实例后再一键配齐工作流包。",
                "Please wake or connect to a cloud G4 instance on the left before installing a workflow pack."
            )
            return
        }
        comfyDownloadingWeight = true
        comfyInstallingPackId = pack.id
        comfyProgress = 0.04
        let tip = L(
            "正在为您一键配齐「\(pack.localizedTitle)」模型权重与全中文工作流...",
            "Installing models & workflow templates for \"\(pack.localizedTitle)\"..."
        )
        comfyStatusMessage = tip
        statusMessage = tip

        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 30
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "pack_id": pack.id,
                "auto_start": true,
                "hf_token": hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
            ])
            let (data, resp) = try await adminSession.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            if code == 404 {
                comfyDownloadingWeight = false
                comfyInstallingPackId = ""
                let upgradeTip = L(
                    "当前云端实例仍在运行旧版服务，请点左侧「唤醒实例并部署」热更新服务端（不会释放现有 G4 实例与画布），随后即可一键配齐工作流！",
                    "The cloud instance is running an older server version. Click 'Wake & Deploy' on the left to hot-update app.py without releasing your G4 VM, then install the pack!"
                )
                comfyStatusMessage = upgradeTip
                statusMessage = upgradeTip
                return
            }
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)

            for _ in 0..<600 {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await checkColabStatus(silent: true)
                statusMessage = comfyStatusMessage
                if !comfyDownloadingWeight && comfyInstallingPackId.isEmpty && !comfyStarting {
                    break
                }
            }
            if openWhenReady && comfyRunning && !comfyPublicURL.isEmpty {
                openComfyUIInBrowser()
            }
        } catch {
            comfyDownloadingWeight = false
            comfyInstallingPackId = ""
            statusMessage = L(
                "一键配齐预置工作流包失败: \(error.localizedDescription)",
                "Failed to install preset workflow pack: \(error.localizedDescription)"
            )
        }
    }

    func downloadComfyOutputToMac(_ item: ComfyOutputMediaItem) async {
        let base = normalizedBaseURL
        let encodedRel = item.rel_path
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/output_media/\(encodedRel)") else { return }
        comfyPullingOutputRelPath = item.rel_path
        statusMessage = L(
            "正在从 ComfyUI 云端输出目录拉取 \(item.filename) 到本地 Mac...",
            "Pulling \(item.filename) from ComfyUI cloud output folder to Mac..."
        )
        defer { comfyPullingOutputRelPath = nil }
        do {
            let (tmpURL, resp) = try await adminSession.download(for: adminRequest(url))
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                statusMessage = L(
                    "拉取成片失败: 云端文件不存在或尚未写入完成",
                    "Failed to pull output: remote file does not exist or is still being written"
                )
                return
            }
            let safeName = "comfy_" + (item.filename as NSString).lastPathComponent
            let dstURL = localSaveDir.appendingPathComponent(safeName)
            if FileManager.default.fileExists(atPath: dstURL.path) {
                try? FileManager.default.removeItem(at: dstURL)
            }
            try FileManager.default.moveItem(at: tmpURL, to: dstURL)
            refreshLocalMedia()
            let ext = dstURL.pathExtension.lowercased()
            currentMediaURL = dstURL
            if ["png", "jpg", "jpeg", "webp", "gif"].contains(ext), let img = NSImage(contentsOf: dstURL) {
                player?.pause()
                player = nil
                currentPreviewImage = img
            } else {
                currentPreviewImage = nil
                player = AVPlayer(url: dstURL)
                player?.play()
            }
            statusMessage = L(
                "已将 ComfyUI 成片保存至本地目录: \(dstURL.lastPathComponent)",
                "Saved ComfyUI output to local folder: \(dstURL.lastPathComponent)"
            )
        } catch {
            statusMessage = L(
                "下载 ComfyUI 成片失败: \(error.localizedDescription)",
                "Failed to download ComfyUI output: \(error.localizedDescription)"
            )
        }
    }

    func regenerateApiKey() {
        apiAccessKey = "sk-colab-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(20).lowercased()
        Task { await syncApiKeyToColab() }
    }

    func syncApiKeyToColab(initializeOnly: Bool = false) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/config_api_key") else { return }
        let payloadKey = apiAccessKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !payloadKey.isEmpty else { return }
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["api_key": payloadKey, "initialize_only": initializeOnly])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            remoteApiKeySynced = true
            applyStatusResponse(decoded)
        } catch {}
    }

    func syncContextLengthToColab(announce: Bool = false) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/config_context") else { return }
        let targetTokens = contextLengthTokens
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["context_length": targetTokens])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            remoteContextSynced = true
            applyStatusResponse(decoded)
            if announce, let msg = decoded.task_message, !msg.isEmpty {
                statusMessage = msg
            }
        } catch {}
    }

}
