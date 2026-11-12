import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers
import ObjectiveC.runtime

// MARK: - 核心视图模型 (ViewModel)

final class SameOriginAdminRedirect: NSObject, URLSessionTaskDelegate {
    static func permits(_ target: URL?, from origin: URL?) -> Bool {
        guard let target, let origin, let host = origin.host, !host.isEmpty,
              target.scheme?.lowercased() == "https", origin.scheme?.lowercased() == "https",
              target.user == nil, target.password == nil, origin.user == nil, origin.password == nil else { return false }
        return target.host?.lowercased() == host.lowercased() && (target.port ?? 443) == (origin.port ?? 443)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        guard let original = task.originalRequest, Self.permits(request.url, from: original.url) else {
            completionHandler(nil)
            return
        }
        var redirected = request
        redirected.setValue(original.value(forHTTPHeaderField: "Authorization"), forHTTPHeaderField: "Authorization")
        completionHandler(redirected)
    }
}

@MainActor
class StudioViewModel: ObservableObject {
    var cachedBridgePythonBin: String? = nil

    var bridgeScript: String { LocalRuntimePaths.resolveBundledOrRepoFile("colab_bridge.py") }
    var appPyPath: String { LocalRuntimePaths.resolveBundledOrRepoFile("app.py") }

    @Published var maskServerURL: Bool = UserDefaults.standard.bool(forKey: "maskServerURL") {
        didSet { UserDefaults.standard.set(maskServerURL, forKey: "maskServerURL") }
    }

    func maskedURLDisplay(_ rawURL: String) -> String {
        let trimmed = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard maskServerURL, !trimmed.isEmpty else { return trimmed }
        if let u = URL(string: trimmed), let scheme = u.scheme, let host = u.host {
            let parts = host.split(separator: ".")
            let suffix = parts.count >= 2 ? parts.suffix(2).joined(separator: ".") : host
            return "\(scheme)://••••••••••••.\(suffix)"
        }
        return "https://••••••••••••"
    }

    static func normalizeServerURL(_ rawURL: String) -> String {
        var raw = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
        while raw.hasSuffix("/") { raw.removeLast() }
        return raw
    }

    private static func savedAdminPassword(for address: String) -> String {
        let normalized = normalizeServerURL(address)
        guard !normalized.isEmpty else { return "" }
        let passwords = UserDefaults.standard.dictionary(forKey: "adminPasswordsByURL") as? [String: String]
        return passwords?[normalized] ?? passwords?[address] ?? ""
    }

    @Published var colabServerURL: String = UserDefaults.standard.string(forKey: "colabServerURL") ?? "" {
        didSet {
            if Self.normalizeServerURL(oldValue) != Self.normalizeServerURL(colabServerURL) {
                adminPassword = Self.savedAdminPassword(for: colabServerURL)
                remoteApiKeySynced = false
                remoteIdleSynced = false
                remoteContextSynced = false
                lastIdleSyncAttemptAt = .distantPast
                lastContextSyncAttemptAt = .distantPast
            }
            UserDefaults.standard.set(colabServerURL, forKey: "colabServerURL")
        }
    }

    @Published var adminPassword = StudioViewModel.savedAdminPassword(
        for: UserDefaults.standard.string(forKey: "colabServerURL") ?? ""
    ) {
        didSet {
            let key = normalizedBaseURL
            guard !key.isEmpty, adminPassword != Self.savedAdminPassword(for: key) else { return }
            var passwords = UserDefaults.standard.dictionary(forKey: "adminPasswordsByURL") as? [String: String] ?? [:]
            passwords[key] = adminPassword.isEmpty ? nil : adminPassword
            if colabServerURL != key {
                passwords.removeValue(forKey: colabServerURL)
            }
            UserDefaults.standard.set(passwords, forKey: "adminPasswordsByURL")
        }
    }
    @Published var hfToken: String = UserDefaults.standard.string(forKey: "hfToken") ?? "" {
        didSet { UserDefaults.standard.set(hfToken, forKey: "hfToken") }
    }

    // 默认初始文案（支持中英文双向无损切换）
    static let defaultGpuNameZh = "未连接 Colab 服务端"
    static let defaultGpuNameEn = "Colab Server Disconnected"
    static let defaultModelNameZh = "未加载"
    static let defaultModelNameEn = "Not Loaded"
    static let defaultComfyStatusZh = "尚未启动云端 ComfyUI 工作台"
    static let defaultComfyStatusEn = "Cloud ComfyUI Studio Not Started"
    static let defaultComfyPersistZh = "Google Drive 自动软链持久化"
    static let defaultComfyPersistEn = "Google Drive Auto-Symlink Persistence"
    static let defaultReadyStatusZh = "就绪：默认已选 G4 (RTX PRO 6000 · 96GB)，可在左侧一键唤醒实例并部署模型。"
    static let defaultReadyStatusEn = "Ready: G4 (RTX PRO 6000 · 96GB) selected by default. Wake & deploy from the left sidebar."
    static let defaultWelcomeLinesZh: [String] = [
        "欢迎使用 EasyColab 多模态云端算力工作台",
        "默认搭载 G4 (RTX PRO 6000 · 96GB)，支持一键部署 ComfyUI 节点工作台 (云盘软链持久化)、MiniMax-H3、Qwen3.8-27B (UD-Q4_K_M GGUF 100 tok/s / BF16)、Qwen-Image-2.1 与 Z-Image-Turbo。"
    ]
    static let defaultWelcomeLinesEn: [String] = [
        "Welcome to EasyColab Multimodal Cloud GPU Studio",
        "Powered by G4 (RTX PRO 6000 · 96GB) by default — 1-click deploy ComfyUI Node Studio (Drive symlink persistence), MiniMax-H3, Qwen3.8-27B (UD-Q4_K_M GGUF 100 tok/s / BF16), Qwen-Image-2.1 & Z-Image-Turbo."
    ]
    static let clearedConsoleLineZh = "已清空历史输出，正在监听云端实时日志..."
    static let clearedConsoleLineEn = "Cleared log history. Listening for live cloud logs..."

    // Colab CLI 唤醒与真实账本状态
    @Published var cliLoggedIn: Bool = false
    @Published var accountEmail: String = ""
    @Published var showOAuthCodeInput: Bool = false
    @Published var oauthCodeText: String = ""
    @Published var officialCUBalance: Double? = nil

    var displayAccountLabel: String {
        guard cliLoggedIn else { return L("未绑定", "Unbound") }
        let raw = accountEmail.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return L("已连接", "Connected") }
        if maskServerURL, let atIdx = raw.firstIndex(of: "@") {
            let prefix = String(raw[..<atIdx].prefix(2))
            let domain = String(raw[atIdx...])
            return "\(prefix)•••\(domain)"
        }
        return raw
    }
    @Published var officialBurnRate: Double? = nil
    @Published var activeVMCount: Int = 0
    @Published var wakeGPUChoice: String = {
        let saved = UserDefaults.standard.string(forKey: "wakeGPUChoiceV2") ?? "G4"
        return ["G4", "H100", "A100", "L4", "T4"].contains(saved) ? saved : "G4"
    }() {
        didSet { UserDefaults.standard.set(wakeGPUChoice, forKey: "wakeGPUChoiceV2") }
    }
    @Published var isWakingColab: Bool = false {
        didSet { syncMacOSRuntimeActivityAssertion() }
    }

    // 远端 GPU 与多模型驻留状态
    @Published var isConnected: Bool = false {
        didSet { syncMacOSRuntimeActivityAssertion() }
    }
    @Published var gpuName: String = L(StudioViewModel.defaultGpuNameZh, StudioViewModel.defaultGpuNameEn) {
        didSet { let c = stripEmoji(gpuName); if c != gpuName { gpuName = c } }
    }
    @Published var vramFreeGB: Double = 0.0
    @Published var vramTotalGB: Double = 0.0
    @Published var vramUsedGB: Double = 0.0
    @Published var ramTotalGB: Double = 0.0
    @Published var ramUsedGB: Double = 0.0
    @Published var ramFreeGB: Double = 0.0
    @Published var diskTotalGB: Double = 0.0
    @Published var diskUsedGB: Double = 0.0
    @Published var diskFreeGB: Double = 0.0
    @Published var shmTotalGB: Double = 0.0
    @Published var shmUsedGB: Double = 0.0
    @Published var shmFreeGB: Double = 0.0
    @Published var driveTotalGB: Double = 0.0
    @Published var driveUsedGB: Double = 0.0
    @Published var driveFreeGB: Double = 0.0
    @Published var vramHistoryPct: [Double] = []
    @Published var ramHistoryPct: [Double] = []
    @Published var diskHistoryPct: [Double] = []
    @Published var combinedTelemetryChartMode: Bool = false
    @Published var selectedModelId: String = "minimax_h3"
    @Published var customModels: [OpenModelSpec] = []
    @Published var showAddCustomModelSheet: Bool = false
    @Published var customRepoInput: String = ""
    @Published var customIncludePatternInput: String = ""
    @Published var customDisplayNameInput: String = ""
    @Published var customModalityInput: String = "auto"
    @Published var isRegisteringCustomModel: Bool = false
    @Published var customModelErrorMessage: String = ""
    @Published var trendingHFModels: [HFTrendingModelItem] = []
    @Published var trendingHFCategory: String = "all"
    @Published var isLoadingTrendingHF: Bool = false
    @Published var activeHFModelDetail: HFModelDetailInfo? = nil
    @Published var isLoadingHFModelDetail: Bool = false
    @Published var showHFModelDetailOverlay: Bool = false
    @Published var loadedModelId: String? = nil
    @Published var loadedModelIds: [String] = []
    @Published var loadedModelsDetail: [LoadedModelSummary] = []
    @Published var modelLoaded: Bool = false
    @Published var isUnloadingModel: Bool = false
    @Published var currentModelName: String = L(StudioViewModel.defaultModelNameZh, StudioViewModel.defaultModelNameEn) {
        didSet { let c = stripEmoji(currentModelName); if c != currentModelName { currentModelName = c } }
    }

    // 算力额度账本
    @Published var totalBudgetCU: Double = {
        let saved = UserDefaults.standard.double(forKey: "totalBudgetCU")
        return saved > 0 ? saved : 2000.0
    }() {
        didSet { UserDefaults.standard.set(totalBudgetCU, forKey: "totalBudgetCU") }
    }
    @Published var uptimeSeconds: Int = 0
    @Published var cuBurnRatePerHour: Double = 0.0
    @Published var sessionCUUsed: Double = 0.0
    @Published var idleTimeoutMinutes: Int = {
        if UserDefaults.standard.object(forKey: "idleTimeoutMinutes") == nil { return 30 }
        let saved = UserDefaults.standard.integer(forKey: "idleTimeoutMinutes")
        return [0, 15, 30, 60].contains(saved) ? saved : 30
    }() {
        didSet {
            UserDefaults.standard.set(idleTimeoutMinutes, forKey: "idleTimeoutMinutes")
        }
    }
    @Published var idleRemainingSeconds: Int = 1800
    var remoteIdleSynced: Bool = false
    @Published var drivePersisted: Bool = false
    @Published var bf16ShardsCount: Int = 0
    @Published var liveConsoleLines: [String] = (
        LanguageManager.shared.isEN ? StudioViewModel.defaultWelcomeLinesEn : StudioViewModel.defaultWelcomeLinesZh
    ) {
        didSet { let c = liveConsoleLines.map(stripEmoji); if c != liveConsoleLines { liveConsoleLines = c } }
    }

    // 当前选中模型的权重明细
    @Published var modelFiles: [ModelFileItem] = OPEN_MODEL_CATALOG[0].defaultFiles
    @Published var downloadedCount: Int = 0
    @Published var totalFileCount: Int = OPEN_MODEL_CATALOG[0].defaultFiles.count
    @Published var allDownloaded: Bool = false
    @Published var totalDiskSizeGB: Double = 0.0
    @Published var isDownloadingWeights: Bool = false
    @Published var isLoadingModel: Bool = false

    // ComfyUI 云端工作台与 Google Drive 软链持久化状态
    @Published var comfyRunning: Bool = false
    @Published var comfyStarting: Bool = false
    @Published var comfySyncingDrive: Bool = false
    @Published var comfyDownloadingWeight: Bool = false
    @Published var comfyInstallingPackId: String = ""
    @Published var comfyPresetPacks: [ComfyPresetPackItem] = DEFAULT_COMFY_PRESET_PACKS
    @Published var comfyProgress: Double = 0.0
    @Published var comfyStatusMessage: String = L(StudioViewModel.defaultComfyStatusZh, StudioViewModel.defaultComfyStatusEn) {
        didSet { let c = stripEmoji(comfyStatusMessage); if c != comfyStatusMessage { comfyStatusMessage = c } }
    }
    @Published var comfyPublicURL: String = ""
    @Published var comfyQueueRunning: Int = 0
    @Published var comfyQueuePending: Int = 0
    @Published var comfyPersistenceLabel: String = L(StudioViewModel.defaultComfyPersistZh, StudioViewModel.defaultComfyPersistEn)
    @Published var comfyDrivePersisted: Bool = false
    @Published var comfyPersistRoot: String = "/content/drive/MyDrive/EasyColab_ComfyUI"
    @Published var comfyCustomNodesCount: Int = 0
    @Published var comfyModelsCount: Int = 0
    @Published var comfyWorkflowsCount: Int = 0
    @Published var comfyRecentOutputs: [ComfyOutputMediaItem] = []
    @Published var comfyUnloadOthersOnStart: Bool = false
    @Published var comfyWeightSourceInput: String = ""
    @Published var comfyWeightFilenameInput: String = ""
    @Published var comfyWeightSubfolder: String = "diffusion_models"
    @Published var comfyPullingOutputRelPath: String? = nil
    @Published var modelTaskProgress: Double = 0.0

    @Published var selectedPrecision: String = "bf16"
    @Published var selectedOffload: String = "disk"

    // 任务模式与生成参数
    @Published var taskModeId: String = "t2va"
    @Published var visualPrompt: String = PRESET_TEMPLATES[0].visual
    @Published var negativePrompt: String = ""
    @Published var soundscapePrompt: String = PRESET_TEMPLATES[0].soundscape
    @Published var musicPrompt: String = PRESET_TEMPLATES[0].music
    @Published var firstFrameImage: NSImage? = nil
    @Published var lastFrameImage: NSImage? = nil
    @Published var selectedVideoResolution: ResolutionOption = VIDEO_RESOLUTIONS[0]
    @Published var selectedImageResolution: ResolutionOption = IMAGE_RESOLUTIONS[0]
    @Published var numFrames: Double = 124
    @Published var inferenceSteps: Double = 35
    @Published var fps: Double = 24
    @Published var seed: Int = 42
    @Published var randomizeSeed: Bool = true
    @Published var batchCount: Int = 1
    @Published var denoisingStrength: Double = 0.70

    // Qwen3.8-27B / GGUF 对话与双协议 API 网关专属参数（支持最高 256K 上下文）
    @Published var contextLengthK: Int = {
        let saved = UserDefaults.standard.integer(forKey: "qwenContextLengthK")
        if saved >= 4 && saved <= 256 { return saved }
        return 32
    }() {
        didSet {
            let clamped = contextLengthK.clamped(to: 4...256)
            if contextLengthK != clamped {
                contextLengthK = clamped
                return
            }
            UserDefaults.standard.set(clamped, forKey: "qwenContextLengthK")
        }
    }
    @Published var loadedContextLengthTokens: Int = 0
    var remoteContextSynced: Bool = false

    var contextLengthTokens: Int {
        contextLengthK.clamped(to: 4...256) * 1024
    }

    @Published var enableThinking: Bool = true {
        didSet {
            // 自动对齐 Unsloth Qwen3.8 官方推荐混合思考采样表
            if enableThinking {
                temperature = 1.0
                topP = 0.95
                presencePenalty = 0.0
            } else {
                temperature = 0.7
                topP = 0.80
                presencePenalty = 1.5
            }
        }
    }
    @Published var reasoningEffort: String = {
        let saved = UserDefaults.standard.string(forKey: "reasoningEffort") ?? "medium"
        return ["low", "medium", "high", "xhigh"].contains(saved) ? saved : "medium"
    }() {
        didSet {
            UserDefaults.standard.set(reasoningEffort, forKey: "reasoningEffort")
        }
    }
    @Published var maxNewTokens: Double = 16384
    @Published var temperature: Double = 1.0
    @Published var topP: Double = 0.95
    @Published var presencePenalty: Double = 0.0
    @Published var chatHistory: [ChatTurnItem] = []

    let apiKeyEnabled = true
    @Published var apiAccessKey: String = {
        if let saved = UserDefaults.standard.string(forKey: "apiAccessKey"), !saved.isEmpty {
            return saved
        }
        let generated = "sk-colab-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(20).lowercased()
        UserDefaults.standard.set(generated, forKey: "apiAccessKey")
        return generated
    }() {
        didSet {
            UserDefaults.standard.set(apiAccessKey, forKey: "apiAccessKey")
        }
    }
    @Published var apiRequestCount: Int = 0
    @Published var remoteApiKeySynced: Bool = false

    // 任务进度与生成耗时状态
    @Published var isGenerating: Bool = false {
        didSet { syncMacOSRuntimeActivityAssertion() }
    }
    @Published var progressValue: Double = 0.0
    @Published var currentJobElapsed: Double = 0.0
    @Published var lastCompletedElapsed: Double? = nil
    @Published var mediaElapsedMap: [String: Double] = (
        UserDefaults.standard.dictionary(forKey: "mediaElapsedSecondsMap") as? [String: Double]
    ) ?? [:] {
        didSet {
            UserDefaults.standard.set(mediaElapsedMap, forKey: "mediaElapsedSecondsMap")
        }
    }
    @Published var statusMessage: String = L(StudioViewModel.defaultReadyStatusZh, StudioViewModel.defaultReadyStatusEn) {
        didSet { let c = stripEmoji(statusMessage); if c != statusMessage { statusMessage = c } }
    }

    var displayGpuName: String { localizeServerText(gpuName) }
    var displayCurrentModelName: String { localizeServerText(currentModelName) }
    var displayComfyStatusMessage: String { localizeServerText(comfyStatusMessage) }
    var displayComfyPersistenceLabel: String { localizeServerText(comfyPersistenceLabel) }
    var displayStatusMessage: String { localizeServerText(statusMessage) }

    // 多媒体预览与本地历史记录
    @Published var currentMediaURL: URL? = nil
    @Published var currentPreviewImage: NSImage? = nil
    @Published var player: AVPlayer? = nil
    @Published var localMediaItems: [LocalMediaItem] = []
    @Published var showColabGuideSheet: Bool = false
    @Published var guideInitialStep: Int = 0
    @Published var showDisconnectConfirm: Bool = false

    var languageObserver: NSObjectProtocol? = nil

    func toggleLanguage() {
        LanguageManager.shared.toggle()
    }

    func syncLocalizedDynamicTexts() {
        let isEN = LanguageManager.shared.isEN
        if gpuName == Self.defaultGpuNameZh || gpuName == Self.defaultGpuNameEn {
            gpuName = isEN ? Self.defaultGpuNameEn : Self.defaultGpuNameZh
        }
        if currentModelName == Self.defaultModelNameZh || currentModelName == Self.defaultModelNameEn {
            currentModelName = isEN ? Self.defaultModelNameEn : Self.defaultModelNameZh
        }
        if comfyStatusMessage == Self.defaultComfyStatusZh || comfyStatusMessage == Self.defaultComfyStatusEn {
            comfyStatusMessage = isEN ? Self.defaultComfyStatusEn : Self.defaultComfyStatusZh
        }
        if comfyPersistenceLabel == Self.defaultComfyPersistZh || comfyPersistenceLabel == Self.defaultComfyPersistEn {
            comfyPersistenceLabel = isEN ? Self.defaultComfyPersistEn : Self.defaultComfyPersistZh
        }
        if statusMessage == Self.defaultReadyStatusZh || statusMessage == Self.defaultReadyStatusEn {
            statusMessage = isEN ? Self.defaultReadyStatusEn : Self.defaultReadyStatusZh
        }
        if liveConsoleLines == Self.defaultWelcomeLinesZh || liveConsoleLines == Self.defaultWelcomeLinesEn {
            liveConsoleLines = isEN ? Self.defaultWelcomeLinesEn : Self.defaultWelcomeLinesZh
        } else if liveConsoleLines == [Self.clearedConsoleLineZh] || liveConsoleLines == [Self.clearedConsoleLineEn] {
            liveConsoleLines = [isEN ? Self.clearedConsoleLineEn : Self.clearedConsoleLineZh]
        }
        for tpl in PRESET_TEMPLATES where tpl.visualZh != tpl.visualEn {
            if visualPrompt == tpl.visualZh || visualPrompt == tpl.visualEn {
                visualPrompt = tpl.visual
                break
            }
        }
    }

    func openOnboardingGuide(step: Int = 0) {
        guideInitialStep = max(0, min(3, step))
        showColabGuideSheet = true
    }

    var heartbeatTask: Task<Void, Never>? = nil
    var generationTimerTask: Task<Void, Never>? = nil
    var generationStartedAt: Date? = nil
    var consecutiveStatusFailures: Int = 0
    var lastTelemetrySampleDate: Date = .distantPast
    let maxTelemetrySamples: Int = 48

    // macOS 防 App Nap 休眠断言与常驻外部 Runtime Proxy 保活守护进程状态
    var runtimeActivityToken: NSObjectProtocol? = nil
    var keepaliveDaemonProcess: Process? = nil
    var keepaliveDaemonStdinPipe: Pipe? = nil
    var keepaliveDaemonStdoutPipe: Pipe? = nil
    var isStartingKeepaliveDaemon: Bool = false
    @Published var keepaliveProxyWSConnected: Bool = false
    var lastIdleSyncAttemptAt: Date = .distantPast
    var lastContextSyncAttemptAt: Date = .distantPast
    var lastLocalInteractionAt: Date = Date()
    var lastRemoteTouchSentAt: Date = .distantPast
    var localEventMonitor: Any? = nil
    var appTerminateObserver: NSObjectProtocol? = nil

    func syncMacOSRuntimeActivityAssertion() {
        let shouldHold = isConnected || isWakingColab || isGenerating || isDownloadingWeights || isLoadingModel
        if shouldHold {
            if runtimeActivityToken == nil {
                runtimeActivityToken = ProcessInfo.processInfo.beginActivity(
                    options: [.userInitiated],
                    reason: "EasyColab active cloud GPU session keepalive & telemetry"
                )
            }
        } else if let token = runtimeActivityToken {
            ProcessInfo.processInfo.endActivity(token)
            runtimeActivityToken = nil
        }
    }

    func recordTelemetrySample(force: Bool = false) {
        let now = Date()
        if !force && now.timeIntervalSince(lastTelemetrySampleDate) < 0.85 {
            return
        }
        lastTelemetrySampleDate = now

        if vramTotalGB > 0 {
            let pct = min(1.0, max(0.0, vramUsedGB / vramTotalGB))
            if vramHistoryPct.isEmpty {
                vramHistoryPct = Array(repeating: pct, count: 14)
            } else {
                vramHistoryPct.append(pct)
                if vramHistoryPct.count > maxTelemetrySamples {
                    vramHistoryPct.removeFirst(vramHistoryPct.count - maxTelemetrySamples)
                }
            }
        }

        if ramTotalGB > 0 {
            let pct = min(1.0, max(0.0, ramUsedGB / ramTotalGB))
            if ramHistoryPct.isEmpty {
                ramHistoryPct = Array(repeating: pct, count: 14)
            } else {
                ramHistoryPct.append(pct)
                if ramHistoryPct.count > maxTelemetrySamples {
                    ramHistoryPct.removeFirst(ramHistoryPct.count - maxTelemetrySamples)
                }
            }
        }

        if diskTotalGB > 0 {
            let pct = min(1.0, max(0.0, diskUsedGB / diskTotalGB))
            if diskHistoryPct.isEmpty {
                diskHistoryPct = Array(repeating: pct, count: 14)
            } else {
                diskHistoryPct.append(pct)
                if diskHistoryPct.count > maxTelemetrySamples {
                    diskHistoryPct.removeFirst(diskHistoryPct.count - maxTelemetrySamples)
                }
            }
        }
    }

    func applyLiveTelemetrySnapshot(_ snap: LiveTelemetrySnapshot) {
        if let g = snap.gpu_name, !g.isEmpty { gpuName = g }
        if let vf = snap.vram_free_gb { vramFreeGB = vf }
        if let vt = snap.vram_total_gb { vramTotalGB = vt }
        if let vu = snap.vram_used_gb { vramUsedGB = vu }
        if let cr = snap.cu_burn_rate_per_hour { cuBurnRatePerHour = cr }
        if let rt = snap.ram_total_gb, rt > 0 { ramTotalGB = rt }
        if let ru = snap.ram_used_gb { ramUsedGB = ru }
        if let rf = snap.ram_free_gb { ramFreeGB = rf }
        if let dt = snap.disk_total_gb, dt > 0 { diskTotalGB = dt }
        if let du = snap.disk_used_gb { diskUsedGB = du }
        if let df = snap.disk_free_gb { diskFreeGB = df }
        if let st = snap.shm_total_gb { shmTotalGB = st }
        if let su = snap.shm_used_gb { shmUsedGB = su }
        if let sf = snap.shm_free_gb { shmFreeGB = sf }
        if let gt = snap.drive_total_gb, gt > 0 { driveTotalGB = gt }
        if let gu = snap.drive_used_gb { driveUsedGB = gu }
        if let gf = snap.drive_free_gb { driveFreeGB = gf }
        recordTelemetrySample()
    }

    func elapsedForMedia(_ url: URL?) -> Double? {
        guard let url else { return nil }
        return mediaElapsedMap[url.lastPathComponent]
    }

    var activeGenerationToken = UUID()

    func stripConsoleProgressBars() {
        if let sepIdx = liveConsoleLines.firstIndex(where: { $0.contains("实时下载/采样进度流") }) {
            liveConsoleLines = Array(liveConsoleLines.prefix(sepIdx))
        }
    }

    func startLiveGenerationTimer() {
        generationTimerTask?.cancel()
        generationStartedAt = Date()
        currentJobElapsed = 0.0
        generationTimerTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 200_000_000)
                guard let self, self.isGenerating, let started = self.generationStartedAt else { break }
                let localElapsed = Date().timeIntervalSince(started)
                if localElapsed > self.currentJobElapsed {
                    self.currentJobElapsed = localElapsed
                }
            }
        }
    }

    func stopLiveGenerationTimer() {
        generationTimerTask?.cancel()
        generationTimerTask = nil
        generationStartedAt = nil
    }

    func resetStaleRemoteRuntimeState(reason: String) {
        let hadActiveState = isConnected
            || !loadedModelsDetail.isEmpty
            || !loadedModelIds.isEmpty
            || modelLoaded
            || isGenerating
            || isLoadingModel
            || isDownloadingWeights
            || vramTotalGB > 0

        stopKeepaliveDaemon()
        stopLiveGenerationTimer()
        isConnected = false
        modelLoaded = false
        loadedModelId = nil
        loadedModelIds = []
        loadedModelsDetail = []
        isGenerating = false
        isLoadingModel = false
        isDownloadingWeights = false
        isUnloadingModel = false
        comfyRunning = false
        comfyStarting = false
        comfySyncingDrive = false
        comfyDownloadingWeight = false
        comfyInstallingPackId = ""
        comfyPublicURL = ""
        comfyQueueRunning = 0
        comfyQueuePending = 0
        progressValue = 0.0
        modelTaskProgress = 0.0
        vramUsedGB = 0.0
        vramFreeGB = 0.0
        vramTotalGB = 0.0
        ramTotalGB = 0.0
        ramUsedGB = 0.0
        ramFreeGB = 0.0
        diskTotalGB = 0.0
        diskUsedGB = 0.0
        diskFreeGB = 0.0
        shmTotalGB = 0.0
        shmUsedGB = 0.0
        shmFreeGB = 0.0
        driveTotalGB = 0.0
        driveUsedGB = 0.0
        driveFreeGB = 0.0
        vramHistoryPct = []
        ramHistoryPct = []
        diskHistoryPct = []
        lastTelemetrySampleDate = .distantPast
        cuBurnRatePerHour = 0.0
        loadedContextLengthTokens = 0
        remoteContextSynced = false
        remoteIdleSynced = false
        remoteApiKeySynced = false
        lastIdleSyncAttemptAt = .distantPast
        lastContextSyncAttemptAt = .distantPast
        syncMacOSRuntimeActivityAssertion()
        gpuName = L(Self.defaultGpuNameZh, Self.defaultGpuNameEn)

        if hadActiveState && !reason.isEmpty {
            statusMessage = reason
        }
    }

    func clearLocalLoadedModelRecord(_ modelId: String) {
        if modelId == "all" {
            loadedModelIds = []
            loadedModelsDetail = []
            loadedModelId = nil
            modelLoaded = false
            loadedContextLengthTokens = 0
        } else {
            loadedModelIds.removeAll { $0 == modelId }
            loadedModelsDetail.removeAll { $0.id == modelId }
            if loadedModelId == modelId {
                loadedModelId = loadedModelIds.first
            }
            modelLoaded = loadedModelIds.contains(selectedModelId)
            if !isQwenLoadedInVRAM {
                loadedContextLengthTokens = 0
            }
        }
    }

    let localSaveDir: URL = {
        let moviesDir = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first!
        let dir = moviesDir.appendingPathComponent("EasyColab", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    let legacySaveDirs: [URL] = {
        let moviesDir = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first!
        return [
            moviesDir.appendingPathComponent("Aevix", isDirectory: true),
            moviesDir.appendingPathComponent("Colab_AI_Studio", isDirectory: true),
            moviesDir.appendingPathComponent("MiniMax_H3_Videos", isDirectory: true)
        ]
    }()

    var allAvailableModels: [OpenModelSpec] {
        OPEN_MODEL_CATALOG + customModels
    }

    var activeModelSpec: OpenModelSpec {
        allAvailableModels.first(where: { $0.id == selectedModelId }) ?? OPEN_MODEL_CATALOG[0]
    }

    var activeModeItem: TaskModeItem {
        activeModelSpec.modes.first(where: { $0.id == taskModeId }) ?? activeModelSpec.modes[0]
    }

    var isSelectedModelLoadedInVRAM: Bool {
        loadedModelIds.contains(selectedModelId) || (modelLoaded && loadedModelId == selectedModelId)
    }

    var isQwenLoadedInVRAM: Bool {
        loadedModelIds.contains("qwen38_27b")
            || loadedModelIds.contains("qwen38_27b_gguf")
            || (modelLoaded && (loadedModelId == "qwen38_27b" || loadedModelId == "qwen38_27b_gguf"))
            || customModels.contains(where: { $0.modality == .llmVLM && loadedModelIds.contains($0.id) })
    }

    let customModelsDefaultsKey = "easycolab.macos.custom_models.v1"

    init() {
        loadPersistedCustomModels()
        refreshLocalMedia()
        startBackgroundTelemetryLoop()
        localEventMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.keyDown, .leftMouseDown, .rightMouseDown, .scrollWheel]
        ) { [weak self] event in
            self?.lastLocalInteractionAt = Date()
            return event
        }
        appTerminateObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.stopKeepaliveDaemon()
            }
        }
        languageObserver = NotificationCenter.default.addObserver(
            forName: .easyColabLanguageDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.syncLocalizedDynamicTexts()
                self.objectWillChange.send()
            }
        }
        Task {
            await refreshOfficialColabAccountUsage()
        }
    }

    deinit {
        if let localEventMonitor {
            NSEvent.removeMonitor(localEventMonitor)
        }
        if let appTerminateObserver {
            NotificationCenter.default.removeObserver(appTerminateObserver)
        }
        if let languageObserver {
            NotificationCenter.default.removeObserver(languageObserver)
        }
    }

    func switchSelectedModel(_ spec: OpenModelSpec) {
        guard selectedModelId != spec.id else { return }
        selectedModelId = spec.id
        taskModeId = spec.modes[0].id
        inferenceSteps = spec.defaultSteps
        selectedPrecision = "bf16"
        modelFiles = spec.defaultFiles
        downloadedCount = 0
        totalFileCount = spec.defaultFiles.count
        allDownloaded = false
        totalDiskSizeGB = 0.0
        modelLoaded = loadedModelIds.contains(spec.id) || (loadedModelId == spec.id)

        // 自动填入匹配该模态的首条示例提示词（若当前框还是另一模态的默认模板）
        if let sample = PRESET_TEMPLATES.first(where: {
            $0.category == spec.modality || (spec.modality == .video && $0.category == .videoAudio)
        }) {
            visualPrompt = sample.visual
            soundscapePrompt = sample.soundscape
            musicPrompt = sample.music
        }

        if spec.modality == .comfyUI {
            if isConnected {
                statusMessage = comfyRunning && !comfyPublicURL.isEmpty
                    ? L("ComfyUI 云端节点工作台正在运行，可一键在浏览器打开全屏画布。", "ComfyUI Cloud Node Studio is running. Open the fullscreen canvas in your browser.")
                    : L("已切换至 ComfyUI 节点工作台，点击「一键拉起云端 ComfyUI」即可挂载 Google Drive 软链并开通画布。", "Switched to ComfyUI Node Studio. Click 'Deploy & Open Cloud ComfyUI' to mount Google Drive symlinks and launch the canvas.")
            } else {
                statusMessage = L("已切换至 ComfyUI 节点工作台，唤醒并连接 G4 实例后即可一键部署并持久化到 Google Drive。", "Switched to ComfyUI Node Studio. Wake & connect a G4 instance to deploy and persist to Google Drive.")
            }
            return
        }

        if isConnected {
            Task {
                await notifyRemoteSelectModel(spec.id)
            }
        } else {
            statusMessage = L(
                "已切换至 \(spec.name) (\(spec.repoId))，连接 Colab 后即可一键下载并加载至显存。",
                "Switched to \(spec.name) (\(spec.repoId)). Connect to Colab to download & load into VRAM."
            )
        }
    }

    func notifyRemoteSelectModel(_ modelId: String) async {
        let base = normalizedBaseURL
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/select_model") else { return }
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["model_id": modelId])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
            if isSelectedModelLoadedInVRAM {
                statusMessage = L(
                    "\(activeModelSpec.name) 已驻留 GPU 显存，可直接开始推理！",
                    "\(activeModelSpec.name) is resident in GPU VRAM and ready for inference!"
                )
            } else if allDownloaded {
                statusMessage = L(
                    "\(activeModelSpec.name) 权重已在磁盘就绪，请点击左侧「加载到显存」。",
                    "\(activeModelSpec.name) weights are ready on disk. Click 'Load to VRAM' on the left."
                )
            } else {
                statusMessage = L(
                    "已切换至 \(activeModelSpec.name) | 权重进度 \(downloadedCount)/\(totalFileCount)，请先点击「下载权重」。",
                    "Switched to \(activeModelSpec.name) | Weights \(downloadedCount)/\(totalFileCount). Click 'Download Weights' first."
                )
            }
        } catch {}
    }

    let adminSession = URLSession(configuration: .ephemeral, delegate: SameOriginAdminRedirect(), delegateQueue: nil)

    func adminRequest(_ url: URL) throws -> URLRequest {
        guard SameOriginAdminRedirect.permits(url, from: URL(string: normalizedBaseURL)),
              !adminPassword.isEmpty else { throw URLError(.userAuthenticationRequired) }
        var request = URLRequest(url: url)
        let encoded = Data("easycolab:\(adminPassword)".utf8).base64EncodedString()
        request.setValue("Basic " + encoded, forHTTPHeaderField: "Authorization")
        return request
    }

    var normalizedBaseURL: String {
        Self.normalizeServerURL(colabServerURL)
    }

    var formattedUptime: String {
        let h = uptimeSeconds / 3600
        let m = (uptimeSeconds % 3600) / 60
        let s = uptimeSeconds % 60
        return String(format: "%02d:%02d:%02d", h, m, s)
    }

    var formattedIdleRemaining: String {
        if idleTimeoutMinutes <= 0 || idleRemainingSeconds < 0 {
            return L("已关闭自动休眠", "Auto-sleep disabled")
        }
        let m = idleRemainingSeconds / 60
        let s = idleRemainingSeconds % 60
        return L(
            "\(m)分\(String(format: "%02d", s))秒后自动释放实例",
            "Auto-release in \(m)m \(String(format: "%02d", s))s"
        )
    }

    var displayRemainingCU: Double {
        if let official = officialCUBalance { return official }
        return max(0.0, totalBudgetCU - sessionCUUsed)
    }

    var displayBurnRate: Double {
        if let rate = officialBurnRate, rate > 0 || !isConnected { return rate }
        return cuBurnRatePerHour
    }


    func checkColabStatus(silent: Bool = false) async {
        let base = normalizedBaseURL
        let now = Date()
        let shouldTouchActivity = !silent || (
            now.timeIntervalSince(lastLocalInteractionAt) < 30.0
            && now.timeIntervalSince(lastRemoteTouchSentAt) >= 20.0
        )
        let touchSuffix = shouldTouchActivity ? "&touch=1" : ""
        guard !base.isEmpty,
              let url = URL(string: "\(base)/minimax_api/status?model_id=\(selectedModelId)\(touchSuffix)") else {
            if !silent {
                statusMessage = L(
                    "请先填入服务地址，或点击左侧「唤醒实例并部署」。",
                    "Please enter a Server URL, or click 'Wake & Deploy' on the left."
                )
            }
            isConnected = false
            return
        }
        if shouldTouchActivity {
            lastRemoteTouchSentAt = now
        }

        if !silent {
            statusMessage = L(
                "正在同步 Colab GPU 显存与 \(activeModelSpec.name) 权重状态...",
                "Syncing Colab GPU VRAM & \(activeModelSpec.name) weight status..."
            )
        }
        do {
            var req = try adminRequest(url)
            req.timeoutInterval = isGenerating ? 28 : (silent ? 20 : 22)
            let (data, resp) = try await adminSession.data(for: req)
            let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? 0

            if statusCode == 404 {
                // 生成视频高负载期间单次 404 可能为 Gradio FRP 隧道瞬断，需连续多次确认才判定进程已停止
                consecutiveStatusFailures += (isGenerating ? 1 : 2)
                if !silent || consecutiveStatusFailures >= 6 {
                    consecutiveStatusFailures = 0
                    resetStaleRemoteRuntimeState(
                        reason: L(
                            "云端 Colab 推理服务已停止（可能因显存溢出或实例释放），已自动重置本地显存状态；请点击左侧「唤醒实例并部署」重新启动。",
                            "Cloud Colab server has stopped (OOM or runtime released). Local VRAM state reset; click 'Wake & Deploy' on the left to restart."
                        )
                    )
                }
                return
            }

            guard statusCode == 200 else {
                consecutiveStatusFailures += 1
                let failureLimit = isGenerating ? 8 : 5
                if !silent || consecutiveStatusFailures >= failureLimit {
                    resetStaleRemoteRuntimeState(
                        reason: L(
                            "无法连接至 Colab 服务端 (HTTP \(statusCode))，实例可能已关闭或网络中断。",
                            "Unable to reach Colab server (HTTP \(statusCode)). Runtime may be stopped or network disconnected."
                        )
                    )
                }
                return
            }

            consecutiveStatusFailures = 0
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)

            if !silent {
                if isSelectedModelLoadedInVRAM {
                    statusMessage = L(
                        "已连接 \(gpuName) | \(activeModelSpec.name) 已驻留显存，可立即开始生成！",
                        "Connected to \(gpuName) | \(activeModelSpec.name) is in VRAM and ready to generate!"
                    )
                } else if allDownloaded {
                    statusMessage = L(
                        "已连接 \(gpuName) | \(activeModelSpec.name) 权重已就绪 (\(String(format: "%.1f", totalDiskSizeGB)) GB)，请点击「加载到显存」。",
                        "Connected to \(gpuName) | \(activeModelSpec.name) weights ready (\(String(format: "%.1f", totalDiskSizeGB)) GB). Click 'Load to VRAM'."
                    )
                } else {
                    statusMessage = L(
                        "已连接 \(gpuName) | \(activeModelSpec.name) 权重进度 \(downloadedCount)/\(totalFileCount)，请先点击「下载权重」。",
                        "Connected to \(gpuName) | \(activeModelSpec.name) weights \(downloadedCount)/\(totalFileCount). Click 'Download Weights' first."
                    )
                }
            }
        } catch {
            consecutiveStatusFailures += 1
            let failureLimit = isGenerating ? 8 : 5
            if !silent {
                resetStaleRemoteRuntimeState(reason: L(
                    "连接失败: \(error.localizedDescription)",
                    "Connection failed: \(error.localizedDescription)"
                ))
            } else if consecutiveStatusFailures >= failureLimit && isConnected {
                resetStaleRemoteRuntimeState(
                    reason: L(
                        "与 Colab 服务端连接中断，已重置本地常驻状态；可点击「连接」重试或「唤醒实例并部署」。",
                        "Disconnected from Colab server; local VRAM state reset. Click 'Connect' to retry or 'Wake & Deploy'."
                    )
                )
            }
        }
    }

    func updateIdleTimeoutConfig(_ minutes: Int) async {
        idleTimeoutMinutes = minutes
        remoteIdleSynced = false
        lastIdleSyncAttemptAt = Date()
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/config_idle") else { return }
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["idle_timeout_minutes": minutes])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            if decoded.idle_timeout_minutes == minutes {
                remoteIdleSynced = true
            }
            applyStatusResponse(decoded)
        } catch {}
    }

    func startRemoteDownloadWeights() async {
        let base = normalizedBaseURL
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/download_model") else {
            statusMessage = L("请先连接有效的 Colab 服务端地址。", "Please connect to a valid Colab Server URL first.")
            return
        }

        let targetModel = selectedModelId
        isDownloadingWeights = true
        modelTaskProgress = 0.05
        statusMessage = L(
            "已通知 Colab 启动 8 线程并发下载 \(activeModelSpec.name)...",
            "Started 8-thread parallel weight download on Colab for \(activeModelSpec.name)..."
        )

        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "model_id": targetModel,
                "hf_token": hfToken,
                "repo_id": activeModelSpec.isCustom ? activeModelSpec.repoId : "",
                "include_pattern": activeModelSpec.isCustom ? activeModelSpec.includePattern : "",
                "custom_name": activeModelSpec.isCustom ? activeModelSpec.nameZh : ""
            ])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)

            while isDownloadingWeights {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                if let statusURL = URL(string: "\(base)/minimax_api/status?model_id=\(targetModel)"),
                   let (sData, _) = try? await adminSession.data(for: adminRequest(statusURL)),
                   let sDecoded = try? JSONDecoder().decode(ColabStatusResponse.self, from: sData) {
                    applyStatusResponse(sDecoded)
                    if let msg = sDecoded.task_message, !msg.isEmpty { statusMessage = msg }
                    if !(sDecoded.is_downloading ?? false) {
                        isDownloadingWeights = false
                        break
                    }
                }
            }
        } catch {
            isDownloadingWeights = false
            statusMessage = L(
                "下载权重请求失败: \(error.localizedDescription)",
                "Weight download request failed: \(error.localizedDescription)"
            )
        }
    }

    func loadRemoteModelToVRAM() async {
        let base = normalizedBaseURL
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/load_model") else {
            statusMessage = L("请先填入有效的 Colab 服务端地址。", "Please enter a valid Colab Server URL first.")
            return
        }

        if !allDownloaded {
            statusMessage = L(
                "\(activeModelSpec.name) 权重尚未下载完整 (\(downloadedCount)/\(totalFileCount))，请先点击「下载权重」。",
                "\(activeModelSpec.name) weights are incomplete (\(downloadedCount)/\(totalFileCount)). Click 'Download Weights' first."
            )
            return
        }

        let targetModel = selectedModelId
        isLoadingModel = true
        modelTaskProgress = 0.15
        if activeModelSpec.modality == .llmVLM {
            statusMessage = L(
                "正在将 \(activeModelSpec.name) 装载进 Colab GPU 显存 (上下文 \(contextLengthK)K)...",
                "Loading \(activeModelSpec.name) into Colab GPU VRAM (Context \(contextLengthK)K)..."
            )
        } else {
            statusMessage = L(
                "正在将 \(activeModelSpec.name) 装载进 Colab GPU 显存...",
                "Loading \(activeModelSpec.name) into Colab GPU VRAM..."
            )
        }

        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let body: [String: Any] = [
                "model_id": targetModel,
                "precision_mode": "bf16",
                "offload_mode": selectedOffload,
                "context_length": contextLengthTokens,
                "hf_token": hfToken,
                "repo_id": activeModelSpec.isCustom ? activeModelSpec.repoId : "",
                "include_pattern": activeModelSpec.isCustom ? activeModelSpec.includePattern : "",
                "custom_name": activeModelSpec.isCustom ? activeModelSpec.nameZh : ""
            ]
            req.httpBody = try JSONSerialization.data(withJSONObject: body)

            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)

            while isLoadingModel {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                if let statusURL = URL(string: "\(base)/minimax_api/status?model_id=\(targetModel)"),
                   let (sData, _) = try? await adminSession.data(for: adminRequest(statusURL)),
                   let sDecoded = try? JSONDecoder().decode(ColabStatusResponse.self, from: sData) {
                    applyStatusResponse(sDecoded)
                    if let msg = sDecoded.task_message, !msg.isEmpty { statusMessage = msg }
                    if !(sDecoded.is_loading_vram ?? false) {
                        isLoadingModel = false
                        break
                    }
                }
            }
        } catch {
            isLoadingModel = false
            statusMessage = L(
                "加载模型请求失败: \(error.localizedDescription)",
                "Load model request failed: \(error.localizedDescription)"
            )
        }
    }

    func unloadRemoteModelFromVRAM(modelId: String) async {
        let targetLabel = modelId == "all"
            ? L("全部常驻模型", "all resident models")
            : (allAvailableModels.first(where: { $0.id == modelId })?.name ?? modelId)

        if modelId == "all" || modelId == selectedModelId {
            isGenerating = false
            progressValue = 0.0
        }

        let base = normalizedBaseURL
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/unload_model") else {
            clearLocalLoadedModelRecord(modelId)
            statusMessage = L(
                "已清理本地 \(targetLabel) 常驻记录。",
                "Cleared local VRAM residency record for \(targetLabel)."
            )
            return
        }

        isUnloadingModel = true
        defer { isUnloadingModel = false }
        statusMessage = L(
            "正在中断关联任务并从 GPU 显存卸载 \(targetLabel)...",
            "Interrupting related tasks and unloading \(targetLabel) from GPU VRAM..."
        )

        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 20
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["model_id": modelId])
            let (data, resp) = try await adminSession.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            if code == 200 {
                let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
                applyStatusResponse(decoded)
                if let msg = decoded.task_message, !msg.isEmpty {
                    statusMessage = msg
                } else {
                    statusMessage = L(
                        "已从显存卸载 \(targetLabel)，当前可用显存 \(String(format: "%.1f", vramFreeGB)) GB。",
                        "Unloaded \(targetLabel) from VRAM. Free VRAM: \(String(format: "%.1f", vramFreeGB)) GB."
                    )
                }
            } else {
                // 远端已离线或 404，直接清理本地残留状态，绝不卡死按钮
                resetStaleRemoteRuntimeState(reason: L(
                    "远端 Colab 服务已停止，已清理本地 \(targetLabel) 显存记录。",
                    "Remote Colab server stopped; cleared local VRAM record for \(targetLabel)."
                ))
            }
        } catch {
            clearLocalLoadedModelRecord(modelId)
            if !isConnected {
                resetStaleRemoteRuntimeState(reason: L(
                    "远端实例未连接，已移除本地 \(targetLabel) 常驻记录。",
                    "Remote instance disconnected; removed local VRAM record for \(targetLabel)."
                ))
            } else {
                statusMessage = L(
                    "远端响应超时，已先移除本地 \(targetLabel) 常驻状态 (\(error.localizedDescription))。",
                    "Remote timed out; cleared local VRAM record for \(targetLabel) (\(error.localizedDescription))."
                )
            }
        }
    }

    func cancelCurrentGeneration() async {
        activeGenerationToken = UUID()
        stopLiveGenerationTimer()
        isGenerating = false
        progressValue = 0.0
        currentJobElapsed = 0.0
        stripConsoleProgressBars()
        statusMessage = L(
            "已发送停止指令，正在中断云端 GPU 推理任务...",
            "Stop signal sent. Aborting cloud GPU inference..."
        )

        let base = normalizedBaseURL
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/cancel") else { return }
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 12
            let (data, resp) = try await adminSession.data(for: req)
            if (resp as? HTTPURLResponse)?.statusCode == 200,
               let decoded = try? JSONDecoder().decode(ColabStatusResponse.self, from: data) {
                applyStatusResponse(decoded)
            }
            stripConsoleProgressBars()
            statusMessage = L(
                "已停止当前推理任务，GPU 已恢复就绪。",
                "Stopped current inference task. GPU is ready."
            )
        } catch {
            statusMessage = L(
                "本地已停止；远端连接异常: \(error.localizedDescription)",
                "Stopped locally; remote connection error: \(error.localizedDescription)"
            )
        }
    }

    func startGeneration() async {
        let promptTrimmed = visualPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !promptTrimmed.isEmpty else {
            statusMessage = activeModelSpec.modality == .llmVLM
                ? L("请输入对话问题或视觉指令！", "Please enter a question or vision prompt!")
                : L("请输入画面描述提示词！", "Please enter a visual scene prompt!")
            return
        }

        if taskModeId == "fl2va" && firstFrameImage == nil && lastFrameImage == nil {
            statusMessage = L(
                "在「首尾帧生视频」模式下，请至少提供一张首帧或尾帧图片。",
                "In First/Last Frame mode, please provide at least a start frame or end frame image."
            )
            return
        }
        if taskModeId == "i2v" && firstFrameImage == nil {
            statusMessage = L(
                "在「首帧图生视频」模式下，请提供一张首帧图片。",
                "In Image-to-Video mode, please provide a start frame image."
            )
            return
        }
        if taskModeId == "i2i_edit" && firstFrameImage == nil && lastFrameImage == nil {
            statusMessage = L(
                "在「图像编辑/重绘」模式下，请至少拖入一张参考原图。",
                "In Image Edit / Restyle mode, please drop at least one reference image."
            )
            return
        }

        await checkColabStatus(silent: true)
        guard isConnected else {
            statusMessage = L(
                "尚未连接 Colab 服务端，请先唤醒或连接现有实例。",
                "Not connected to Colab server. Please wake & deploy or connect to an active runtime first."
            )
            return
        }
        guard isSelectedModelLoadedInVRAM else {
            if !allDownloaded {
                statusMessage = L(
                    "请先在左侧点击「下载权重」，再点击「加载到显存」装载 \(activeModelSpec.name)。",
                    "Click 'Download Weights' first, then 'Load to VRAM' to load \(activeModelSpec.name)."
                )
            } else {
                statusMessage = L(
                    "\(activeModelSpec.name) 尚未装载至显存，请先点击左侧「加载到显存」。",
                    "\(activeModelSpec.name) is not in VRAM yet. Click 'Load to VRAM' on the left first."
                )
            }
            return
        }

        let base = normalizedBaseURL
        guard let submitURL = URL(string: "\(base)/minimax_api/generate") else { return }

        let myToken = UUID()
        activeGenerationToken = myToken
        isGenerating = true
        progressValue = 0.0
        currentJobElapsed = 0.0
        stripConsoleProgressBars()
        startLiveGenerationTimer()
        statusMessage = L(
            "正在向 Colab 提交 \(activeModelSpec.name) 推理任务 (0%)...",
            "Submitting \(activeModelSpec.name) inference job to Colab (0%)..."
        )

        let resOpt = activeModelSpec.modality == .image ? selectedImageResolution : selectedVideoResolution

        var chatPayload: [[String: String]] = []
        var assistantRowIndex: Int? = nil
        if activeModelSpec.modality == .llmVLM {
            let userTurn = ChatTurnItem(
                role: "user",
                content: promptTrimmed,
                hasImage: (taskModeId == "vlm" && (firstFrameImage != nil || lastFrameImage != nil))
            )
            chatHistory.append(userTurn)
            chatPayload = chatHistory.map { ["role": $0.role, "content": $0.content] }
            let placeholderAssistant = ChatTurnItem(role: "assistant", content: "", thinking: "")
            chatHistory.append(placeholderAssistant)
            assistantRowIndex = chatHistory.count - 1
        }

        do {
            var req = try adminRequest(submitURL)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.timeoutInterval = 30

            var payload: [String: Any] = [
                "model_id": selectedModelId,
                "task_mode": taskModeId,
                "visual_prompt": promptTrimmed,
                "negative_prompt": negativePrompt,
                "soundscape_prompt": soundscapePrompt,
                "music_prompt": musicPrompt,
                "width": resOpt.width,
                "height": resOpt.height,
                "num_frames": Int(numFrames),
                "num_inference_steps": Int(inferenceSteps),
                "fps": Int(fps),
                "seed": seed,
                "randomize_seed": randomizeSeed,
                "batch_count": activeModelSpec.modality == .llmVLM ? 1 : batchCount,
                "denoising_strength": denoisingStrength,
                "enable_thinking": enableThinking,
                "max_new_tokens": Int(maxNewTokens),
                "context_length": contextLengthTokens,
                "temperature": temperature,
                "top_p": topP,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": presencePenalty,
                "repeat_penalty": 1.0,
                "reasoning_effort": reasoningEffort
            ]
            if !chatPayload.isEmpty {
                payload["chat_messages"] = chatPayload
            }

            let needsImage = ["fl2va", "i2v", "i2i_edit", "vlm"].contains(taskModeId)
            if needsImage {
                if let firstImg = firstFrameImage, let b64 = encodeImageToBase64(firstImg) {
                    payload["first_frame_base64"] = b64
                }
                if let lastImg = lastFrameImage, let b64 = encodeImageToBase64(lastImg) {
                    payload["last_frame_base64"] = b64
                }
            }

            req.httpBody = try JSONSerialization.data(withJSONObject: payload)
            let (data, _) = try await adminSession.data(for: req)
            guard activeGenerationToken == myToken else { return }
            let job = try JSONDecoder().decode(ColabJobResponse.self, from: data)

            if job.status == "failed" {
                stopLiveGenerationTimer()
                isGenerating = false
                progressValue = 0.0
                statusMessage = job.message
                if let idx = assistantRowIndex, idx < chatHistory.count {
                    chatHistory[idx].content = L(
                        "推理失败：\(job.message)",
                        "Inference failed: \(localizeServerText(job.message))"
                    )
                }
                return
            }

            await pollColabJob(baseURL: base, jobId: job.job_id, assistantRowIndex: assistantRowIndex, token: myToken)

        } catch {
            guard activeGenerationToken == myToken else { return }
            stopLiveGenerationTimer()
            isGenerating = false
            progressValue = 0.0
            statusMessage = L(
                "提交任务失败: \(error.localizedDescription)",
                "Failed to submit job: \(error.localizedDescription)"
            )
        }
    }

    func pollColabJob(baseURL: String, jobId: String, assistantRowIndex: Int?, token: UUID) async {
        guard let pollURL = URL(string: "\(baseURL)/minimax_api/job/\(jobId)") else {
            if activeGenerationToken == token {
                stopLiveGenerationTimer()
                isGenerating = false
            }
            return
        }

        var downloadedFilenames = Set<String>()
        var consecutivePollErrors = 0
        var consecutiveNotFound = 0
        let pollIntervalNs: UInt64 = assistantRowIndex != nil ? 600_000_000 : 1_000_000_000

        while isGenerating && activeGenerationToken == token {
            try? await Task.sleep(nanoseconds: pollIntervalNs)
            guard isGenerating && activeGenerationToken == token else { break }
            do {
                var req = try adminRequest(pollURL)
                req.timeoutInterval = 30
                let (data, resp) = try await adminSession.data(for: req)
                guard isGenerating && activeGenerationToken == token else { break }
                let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                if code == 404 {
                    consecutiveNotFound += 1
                    if consecutiveNotFound >= 5 {
                        resetStaleRemoteRuntimeState(
                            reason: L(
                                "云端推理进程已中断（可能因显存溢出或实例释放），请重新唤醒实例。",
                                "Cloud inference process interrupted (OOM or runtime released). Please wake & deploy again."
                            )
                        )
                        break
                    }
                    continue
                }
                consecutiveNotFound = 0
                guard code == 200 else {
                    consecutivePollErrors += 1
                    if consecutivePollErrors >= 12 {
                        stopLiveGenerationTimer()
                        isGenerating = false
                        statusMessage = L(
                            "云端任务响应异常 (HTTP \(code))，已自动解除等待。",
                            "Cloud job returned HTTP \(code); unlocked controls automatically."
                        )
                        break
                    }
                    continue
                }
                consecutivePollErrors = 0
                let job = try JSONDecoder().decode(ColabJobResponse.self, from: data)
                if let tl = job.telemetry {
                    applyLiveTelemetrySnapshot(tl)
                }
                // 在文件彻底生成并下载回本地预览前，进度条最高锁定在 98%，杜绝“显示 100% 但视频尚未出来”
                if job.status != "completed" {
                    progressValue = min(0.98, max(0.0, job.progress))
                }
                statusMessage = job.message
                if let remoteEl = job.elapsed, remoteEl >= 0 {
                    currentJobElapsed = max(currentJobElapsed, remoteEl)
                }
                if let s = job.seed { self.seed = s }
                if let lines = job.console_lines, !lines.isEmpty {
                    self.liveConsoleLines = lines
                }

                // Qwen3.8-27B 实时流式更新思考链与回复正文
                if let idx = assistantRowIndex, idx < chatHistory.count {
                    if let thinkTxt = job.thinking_output {
                        chatHistory[idx].thinking = thinkTxt
                    }
                    if let ansTxt = job.text_output {
                        chatHistory[idx].content = ansTxt
                    }
                }

                // 视频或图像批量边生边回传预览
                if let completedItems = job.completed_videos {
                    for item in completedItems where !downloadedFilenames.contains(item.filename) {
                        guard activeGenerationToken == token else { break }
                        if let downloadURL = URL(string: "\(baseURL)\(item.video_url)") {
                            downloadedFilenames.insert(item.filename)
                            if job.status == "completed" && (job.batch_count ?? 1) <= 1 {
                                progressValue = 0.99
                                statusMessage = L(
                                    "云端渲染已完成，正在高速回传结果文件至本地预览 (99%)...",
                                    "Cloud rendering complete. Transferring output to Mac for preview (99%)..."
                                )
                            }
                            await downloadMediaToLocal(
                                from: downloadURL,
                                filename: item.filename,
                                elapsedSeconds: item.elapsed ?? job.elapsed,
                                keepGeneratingProgress: job.status != "completed"
                            )
                        }
                    }
                }

                guard activeGenerationToken == token else { break }

                if job.status == "completed" {
                    let finalElapsed = job.elapsed ?? currentJobElapsed
                    if finalElapsed > 0 {
                        lastCompletedElapsed = finalElapsed
                        currentJobElapsed = finalElapsed
                    }
                    if let fname = job.filename,
                       !downloadedFilenames.contains(fname),
                       let mediaPath = job.video_url,
                       let downloadURL = URL(string: "\(baseURL)\(mediaPath)") {
                        downloadedFilenames.insert(fname)
                        progressValue = 0.99
                        statusMessage = L(
                            "云端渲染已完成，正在高速回传结果文件至本地预览 (99%)...",
                            "Cloud rendering complete. Transferring output to Mac for preview (99%)..."
                        )
                        await downloadMediaToLocal(
                            from: downloadURL,
                            filename: fname,
                            elapsedSeconds: finalElapsed
                        )
                    }
                    guard activeGenerationToken == token else { break }
                    if assistantRowIndex == nil, (job.batch_count ?? 1) > 1, finalElapsed > 0 {
                        let cnt = job.batch_count ?? 1
                        statusMessage = L(
                            "批量生成 \(cnt) 项全部完成 · 总耗时 \(formatDuration(finalElapsed))（均值 \(formatDuration(finalElapsed / Double(cnt)))/项）",
                            "Batch of \(cnt) completed · Total \(formatDuration(finalElapsed)) (avg \(formatDuration(finalElapsed / Double(cnt)))/item)"
                        )
                    }
                    stopLiveGenerationTimer()
                    isGenerating = false
                    progressValue = 1.0
                    await checkColabStatus(silent: true)
                    break
                } else if job.status == "failed" {
                    if let idx = assistantRowIndex, idx < chatHistory.count, chatHistory[idx].content.isEmpty {
                        chatHistory[idx].content = localizeServerText(job.message)
                    }
                    stopLiveGenerationTimer()
                    isGenerating = false
                    progressValue = 0.0
                    stripConsoleProgressBars()
                    break
                }
            } catch {
                guard activeGenerationToken == token else { break }
                consecutivePollErrors += 1
                if consecutivePollErrors >= 12 {
                    stopLiveGenerationTimer()
                    isGenerating = false
                    statusMessage = L(
                        "与云端任务失去连接，已自动解除锁定: \(error.localizedDescription)",
                        "Lost connection to cloud job; unlocked controls automatically: \(error.localizedDescription)"
                    )
                    await checkColabStatus(silent: true)
                    break
                }
                continue
            }
        }
    }

    func downloadMediaToLocal(
        from remoteURL: URL,
        filename: String,
        elapsedSeconds: Double? = nil,
        keepGeneratingProgress: Bool = false
    ) async {
        do {
            let (tempURL, response) = try await adminSession.download(for: adminRequest(remoteURL))
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            let destURL = localSaveDir.appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: destURL.path) {
                try FileManager.default.removeItem(at: destURL)
            }
            try FileManager.default.moveItem(at: tempURL, to: destURL)
            if let el = elapsedSeconds, el > 0 {
                mediaElapsedMap[destURL.lastPathComponent] = el
            }
            refreshLocalMedia()
            selectMedia(destURL)
            if !keepGeneratingProgress {
                progressValue = 1.0
                let isMP4 = destURL.pathExtension.lowercased() == "mp4"
                let kindZh = isMP4 ? "视频" : "图像"
                let kindEn = isMP4 ? "Video" : "Image"
                if let el = elapsedSeconds, el > 0 {
                    statusMessage = L(
                        "\(kindZh)生成完成 · 耗时 \(formatDuration(el)) · 已保存：\(destURL.lastPathComponent)",
                        "\(kindEn) generated in \(formatDuration(el)) · Saved: \(destURL.lastPathComponent)"
                    )
                } else {
                    statusMessage = L(
                        "结果已保存至本地并自动预览：\(destURL.lastPathComponent)",
                        "Saved locally and opened in preview: \(destURL.lastPathComponent)"
                    )
                }
            }
        } catch {
            statusMessage = L(
                "云端生成已完成，但下载至本地失败: \(error.localizedDescription)",
                "Cloud generation finished, but local download failed: \(error.localizedDescription)"
            )
        }
    }

    func disconnectRemoteColab() async {
        stopKeepaliveDaemon()
        statusMessage = L(
            "正在释放云端 Colab 实例并同步账户计费状态...",
            "Terminating cloud Colab instances & syncing billing status..."
        )
        let base = normalizedBaseURL
        if !base.isEmpty, let url = URL(string: "\(base)/minimax_api/disconnect"), var req = try? adminRequest(url) {
            req.httpMethod = "POST"
            req.timeoutInterval = 8
            _ = try? await adminSession.data(for: req)
        }

        if cliLoggedIn {
            let out = await runBridgeCommand(["stop_all"])
            if let data = out.data(using: .utf8),
               let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) {
                officialCUBalance = decoded.paid_balance ?? officialCUBalance
                officialBurnRate = decoded.burn_rate_hourly ?? 0.0
                activeVMCount = decoded.active_assignments ?? 0
                statusMessage = decoded.message ?? L(
                    "已彻底释放全部云端实例，计费速率归零。",
                    "All cloud Colab instances terminated. Billing rate is now 0 CU/h."
                )
            }
        } else {
            statusMessage = L(
                "已发送释放指令，Colab 实例已停止计费。",
                "Release command sent. Colab runtime billing stopped."
            )
        }

        stopKeepaliveDaemon()
        stopLiveGenerationTimer()
        isConnected = false
        modelLoaded = false
        loadedModelId = nil
        loadedModelIds = []
        loadedModelsDetail = []
        loadedContextLengthTokens = 0
        ramTotalGB = 0.0
        ramUsedGB = 0.0
        ramFreeGB = 0.0
        diskTotalGB = 0.0
        diskUsedGB = 0.0
        diskFreeGB = 0.0
        shmTotalGB = 0.0
        shmUsedGB = 0.0
        shmFreeGB = 0.0
        driveTotalGB = 0.0
        driveUsedGB = 0.0
        driveFreeGB = 0.0
        vramHistoryPct = []
        ramHistoryPct = []
        diskHistoryPct = []
        lastTelemetrySampleDate = .distantPast
        remoteContextSynced = false
        remoteIdleSynced = false
        remoteApiKeySynced = false
        lastIdleSyncAttemptAt = .distantPast
        lastContextSyncAttemptAt = .distantPast
        uptimeSeconds = 0
        cuBurnRatePerHour = 0.0
        syncMacOSRuntimeActivityAssertion()
    }

    func encodeImageToBase64(_ image: NSImage) -> String? {
        guard let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let jpegData = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.92]) else {
            return nil
        }
        return jpegData.base64EncodedString()
    }
}

