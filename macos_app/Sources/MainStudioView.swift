import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers
import ObjectiveC.runtime

// MARK: - 主界面视图（左：控制栏 | 中：创作与网关工作台 | 右：独立全高运行日志）

struct MainStudioView: View {
    @StateObject var vm = StudioViewModel()
    @ObservedObject var i18n = LanguageManager.shared
    @Environment(\.colorScheme) var colorScheme
    @State var showFileList: Bool = false
    @State var showNegativePrompt: Bool = false
    @State var openaiSnippetTab: Int = 0
    @State var anthropicSnippetTab: Int = 0
    @State var showRightLogPanel: Bool = true

    var isTaskStreaming: Bool {
        vm.isWakingColab
            || vm.isDownloadingWeights
            || vm.isLoadingModel
            || vm.isUnloadingModel
            || vm.isGenerating
            || vm.comfyStarting
            || vm.comfyDownloadingWeight
            || vm.comfySyncingDrive
            || vm.comfyQueueRunning > 0
    }

    var body: some View {
        let palette = StudioPalette(colorScheme: colorScheme)

        HStack(spacing: 0) {
            // 1. 左侧独立底色控制面板（从窗口最顶部 y=0 贯通到底部）
            VStack(spacing: 0) {
                sidebarTopHeader(palette: palette)
                Rectangle()
                    .fill(palette.splitDivider)
                    .frame(height: 1)
                sidebarPanel(palette: palette)
            }
            .frame(width: 356)
            .background(palette.sidebarBackground)

            // 左-中贯通全高 1pt 锐利分界线
            Rectangle()
                .fill(palette.splitDivider)
                .frame(width: 1)

            // 2. 中间主工作区画布（从窗口最顶部 y=0 贯通到底部）
            VStack(spacing: 0) {
                workspaceTopHeader(palette: palette)
                Rectangle()
                    .fill(palette.splitDivider)
                    .frame(height: 1)
                workspacePanel(palette: palette)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .frame(minWidth: 540, maxWidth: .infinity, maxHeight: .infinity)
            .background(palette.workspaceCanvas)

            // 3. 右侧独立全高运行日志面板（从窗口最顶部 y=0 贯通到底部）
            if showRightLogPanel {
                Rectangle()
                    .fill(palette.splitDivider)
                    .frame(width: 1)

                VStack(spacing: 0) {
                    logPanelTopHeader(palette: palette)
                    Rectangle()
                        .fill(palette.splitDivider)
                        .frame(height: 1)
                    rightLogPanel(palette: palette)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                .frame(width: 352)
                .background(palette.logPanelBackground)
            }
        }
        .id(i18n.lang)
        .ignoresSafeArea(.container, edges: .top)
        .background(TitlebarConfigurator(headerHeight: 40))
        .sheet(isPresented: $vm.showColabGuideSheet) {
            ColabGuideSheet(vm: vm)
        }
        .sheet(isPresented: $vm.showAddCustomModelSheet) {
            AddCustomHFModelMacSheet(vm: vm)
        }
        .onAppear {
            if !UserDefaults.standard.bool(forKey: "hasSeenOnboardingGuide_v1") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    vm.openOnboardingGuide(step: 0)
                }
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .openEasyColabGuide)) { _ in
            vm.openOnboardingGuide(step: 0)
        }
        .alert(L("释放所有 Colab 实例？", "Release All Colab Instances?"), isPresented: $vm.showDisconnectConfirm) {
            Button(L("取消", "Cancel"), role: .cancel) {}
            Button(L("释放", "Release"), role: .destructive) {
                Task { await vm.disconnectRemoteColab() }
            }
        } message: {
            Text(L(
                "账号下所有运行中的 Colab GPU 实例将被销毁，计费立即归零。",
                "All active Colab GPU runtimes under your account will be terminated and CU billing will stop immediately."
            ))
        }
    }

    // MARK: - 一体化无缝顶栏（左中右独立底色 + 1pt 贯通分界线 + 原生拖拽）

    func sidebarTopHeader(palette: StudioPalette) -> some View {
        HStack(spacing: 6) {
            ZStack {
                RoundedRectangle(cornerRadius: 5)
                    .fill(Color(red: 0.88, green: 0.95, blue: 0.98))
                Image(systemName: "cloud.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(.white)
                Canvas { context, size in
                    let points = [CGPoint(x: 5.5, y: 11.7), CGPoint(x: 10, y: 8.5), CGPoint(x: 14.5, y: 11.7)]
                    var line = Path()
                    line.move(to: points[0])
                    line.addLine(to: points[1])
                    line.addLine(to: points[2])
                    context.stroke(line, with: .color(Color(red: 0.98, green: 0.43, blue: 0.19)), style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
                    for point in points {
                        context.fill(Path(ellipseIn: CGRect(x: point.x - 1.15, y: point.y - 1.15, width: 2.3, height: 2.3)), with: .color(Color(red: 1, green: 0.58, blue: 0.27)))
                    }
                }
            }
            .frame(width: 20, height: 20)

            Text("EasyColab")
                .font(.system(size: 13.5, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .fixedSize()

            Spacer(minLength: 4)

            HStack(spacing: 5) {
                Button {
                    vm.toggleLanguage()
                } label: {
                    HStack(spacing: 3) {
                        Image(systemName: "globe")
                            .font(.system(size: 10, weight: .semibold))
                        Text(i18n.isEN ? "中文" : "EN")
                            .font(.system(size: 10.5, weight: .semibold))
                    }
                    .foregroundStyle(.primary.opacity(0.85))
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3.5)
                    .background(
                        Capsule()
                            .fill(Color.primary.opacity(0.06))
                    )
                    .overlay(
                        Capsule()
                            .strokeBorder(palette.cardBorder, lineWidth: 0.8)
                    )
                }
                .buttonStyle(.plain)
                .fixedSize()
                .help(i18n.isEN ? "切换界面为简体中文" : "Switch UI to English")

                Button {
                    vm.openOnboardingGuide(step: 0)
                } label: {
                    HStack(spacing: 3) {
                        Image(systemName: "sparkles")
                            .font(.system(size: 10, weight: .semibold))
                        Text(L("新手引导", "Guide"))
                            .font(.system(size: 10.5, weight: .semibold))
                    }
                    .foregroundStyle(Color.accentColor)
                    .padding(.horizontal, 7.5)
                    .padding(.vertical, 3.5)
                    .background(
                        Capsule()
                            .fill(Color.accentColor.opacity(0.12))
                    )
                    .overlay(
                        Capsule()
                            .strokeBorder(Color.accentColor.opacity(0.32), lineWidth: 0.8)
                    )
                }
                .buttonStyle(.plain)
                .fixedSize()
                .help(L("查看 EasyColab 新手入门引导与分步使用教程", "Open EasyColab quick start guide & walkthrough"))
            }
        }
        .padding(.leading, 76)
        .padding(.trailing, 12)
        .frame(height: 40)
        .background(WindowDragHandle())
    }

    func workspaceTopHeader(palette: StudioPalette) -> some View {
        HStack(spacing: 10) {
            HStack(spacing: 6) {
                StatusDot(color: vm.isConnected ? .green : .secondary.opacity(0.45))
                let ramTag = vm.ramTotalGB > 0
                    ? L(" · 内存 \(fmt(vm.ramUsedGB, 1))/\(fmt(vm.ramTotalGB, 0))G", " · RAM \(fmt(vm.ramUsedGB, 1))/\(fmt(vm.ramTotalGB, 0))G")
                    : ""
                let diskTag = vm.diskTotalGB > 0
                    ? L(" · 磁盘可用 \(fmt(vm.diskFreeGB, 1))/\(fmt(vm.diskTotalGB, 0))G", " · Disk Free \(fmt(vm.diskFreeGB, 1))/\(fmt(vm.diskTotalGB, 0))G")
                    : ""
                Text(vm.isConnected
                     ? L("\(vm.displayGpuName) · 显存 \(fmt(vm.vramUsedGB, 1))/\(fmt(vm.vramTotalGB, 0))G\(ramTag)\(diskTag)",
                         "\(vm.displayGpuName) · VRAM \(fmt(vm.vramUsedGB, 1))/\(fmt(vm.vramTotalGB, 0))G\(ramTag)\(diskTag)")
                     : L("开源多模型云端工作台", "Open Multi-Model Cloud Studio"))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            Menu {
                let matching = PRESET_TEMPLATES.filter {
                    $0.category == vm.activeModelSpec.modality
                        || (vm.activeModelSpec.modality == .video && $0.category == .videoAudio)
                }
                ForEach(matching.isEmpty ? PRESET_TEMPLATES : matching) { tpl in
                    Button(tpl.name) {
                        vm.visualPrompt = tpl.visual
                        if !tpl.soundscape.isEmpty { vm.soundscapePrompt = tpl.soundscape }
                        if !tpl.music.isEmpty { vm.musicPrompt = tpl.music }
                    }
                }
            } label: {
                Label(L("示例模板", "Templates"), systemImage: "text.badge.plus")
                    .font(.system(size: 12, weight: .medium))
            }
            .controlSize(.small)
            .help(L("填入当前模型精选提示词模板", "Insert curated prompt templates for the active model"))

            Button {
                NSWorkspace.shared.open(vm.localSaveDir)
            } label: {
                Label(L("本地输出目录", "Output Folder"), systemImage: "folder")
                    .font(.system(size: 12, weight: .medium))
            }
            .controlSize(.small)
            .help(L("打开本地保存的视频与图像文件夹", "Open local folder for saved videos & images"))

            Button {
                vm.openOnboardingGuide(step: 0)
            } label: {
                Label(L("使用指南", "User Guide"), systemImage: "book.pages")
                    .font(.system(size: 12, weight: .medium))
            }
            .controlSize(.small)
            .help(L("重新打开新手引导页与完整使用指南", "Reopen the interactive onboarding & user guide"))

            Button {
                withAnimation(.easeInOut(duration: 0.18)) {
                    showRightLogPanel.toggle()
                }
            } label: {
                Image(systemName: "sidebar.right")
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(showRightLogPanel ? Color.accentColor : Color.secondary)
            }
            .controlSize(.small)
            .help(showRightLogPanel
                  ? L("收起右侧运行日志面板", "Hide right run log panel")
                  : L("展开右侧运行日志面板", "Show right run log panel"))
        }
        .padding(.horizontal, 18)
        .frame(height: 40)
        .background(
            palette.workspaceCardFill
                .overlay(WindowDragHandle())
        )
    }

    func logPanelTopHeader(palette: StudioPalette) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "terminal")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Color.accentColor)

            Text(L("运行日志", "Run Logs"))
                .font(.system(size: 12.5, weight: .bold))
                .foregroundStyle(.primary)

            HStack(spacing: 4) {
                Circle()
                    .fill(isTaskStreaming ? Color.green : Color.secondary.opacity(0.45))
                    .frame(width: 5.5, height: 5.5)
                Text(isTaskStreaming
                     ? L("实时传输", "Live")
                     : L("\(vm.liveConsoleLines.count) 行", "\(vm.liveConsoleLines.count) lines"))
                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(isTaskStreaming ? Color.green : .secondary)
            }
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(Capsule().fill(Color.primary.opacity(0.06)))

            Spacer()

            Button {
                copyToPasteboard(
                    vm.liveConsoleLines.map(localizeConsoleLine).joined(separator: "\n"),
                    tip: L("已复制全部运行日志到剪贴板", "Copied all run logs to clipboard")
                )
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 11.5, weight: .medium))
            }
            .buttonStyle(.borderless)
            .help(L("复制全部日志", "Copy all logs"))

            Button {
                vm.liveConsoleLines = [L(StudioViewModel.clearedConsoleLineZh, StudioViewModel.clearedConsoleLineEn)]
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 11.5, weight: .medium))
            }
            .buttonStyle(.borderless)
            .help(L("清空当前日志", "Clear console logs"))
        }
        .padding(.horizontal, 14)
        .frame(height: 40)
        .background(
            palette.logHeaderBackground
                .overlay(WindowDragHandle())
        )
    }

    // MARK: - 左侧控制栏（独立冷灰蓝底色 + 悬浮白卡片）

    func sidebarPanel(palette: StudioPalette) -> some View {
        ScrollView {
            VStack(spacing: 12) {
                SidebarCard(
                    title: L("Colab 算力实例", "Colab Compute Instance"),
                    icon: "cpu",
                    badge: vm.isConnected ? L("在线", "Online") : L("离线", "Offline")
                ) {
                    colabConnectionRows
                }

                SidebarCard(
                    title: L("算力额度与自动休眠", "Compute Quota & Auto-Sleep"),
                    icon: "gauge.with.dots.needle.33percent"
                ) {
                    usageRows
                }

                SidebarCard(
                    title: L("开源模型中心", "Open Model Center"),
                    icon: "square.stack.3d.up",
                    badge: vm.loadedModelIds.isEmpty
                        ? "HF \(vm.allAvailableModels.count)"
                        : L("共驻 \(vm.loadedModelIds.count) 模型", "\(vm.loadedModelIds.count) in VRAM")
                ) {
                    openModelHubRows(palette: palette)
                }

                SidebarCard(
                    title: vm.activeModelSpec.modality == .comfyUI
                        ? L("ComfyUI 云盘持久化与控制", "ComfyUI Persistence & Controls")
                        : (vm.activeModelSpec.modality == .llmVLM
                           ? L("推理与对话参数", "Reasoning & Chat Params")
                           : L("规格与采样参数", "Specs & Sampling Params")),
                    icon: vm.activeModelSpec.modality == .comfyUI ? "externaldrive.badge.icloud" : "slider.horizontal.3"
                ) {
                    adaptiveParamRows
                }
            }
            .padding(14)
        }
        .compatScrollIndicatorMargins(.vertical, 6)
    }

    @ViewBuilder
    var colabConnectionRows: some View {
        HStack(spacing: 6) {
            Text(L("Google 账号", "Google Account"))
                .font(.callout)
                .foregroundStyle(.secondary)
            Spacer(minLength: 4)
            Text(vm.displayAccountLabel)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(vm.cliLoggedIn ? Color.green : Color.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
                .help(vm.accountEmail.isEmpty ? L("当前 Google 账号状态", "Google account connection status") : vm.displayAccountLabel)

            if vm.cliLoggedIn {
                Button(L("刷新", "Refresh")) { Task { await vm.refreshOfficialColabAccountUsage() } }
                    .controlSize(.small)

                Menu {
                    Button {
                        Task { await vm.startGoogleOAuthLogin(switchingAccount: true) }
                    } label: {
                        Label(L("切换 Google 账号...", "Switch Google Account..."), systemImage: "arrow.triangle.2.circlepath")
                    }
                    Divider()
                    Button(role: .destructive) {
                        Task { await vm.logoutGoogleAccount() }
                    } label: {
                        Label(L("退出当前账号", "Sign Out"), systemImage: "rectangle.portrait.and.arrow.right")
                    }
                } label: {
                    Text(L("换号", "Account"))
                }
                .controlSize(.small)
                .fixedSize()
                .help(L("切换其他 Google 账号或退出当前绑定", "Switch to another Google account or sign out"))
            } else {
                Button(L("绑定", "Bind")) { Task { await vm.startGoogleOAuthLogin() } }
                    .controlSize(.small)
            }
        }

        if vm.showOAuthCodeInput {
            HStack(spacing: 6) {
                TextField(
                    L("授权码", "Auth Code"),
                    text: $vm.oauthCodeText,
                    prompt: Text(L("粘贴浏览器显示的授权码", "Paste auth code from browser"))
                )
                    .textFieldStyle(.roundedBorder)
                Button(L("确认", "Confirm")) { Task { await vm.submitGoogleOAuthCode() } }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                Button(L("取消", "Cancel")) {
                    vm.showOAuthCodeInput = false
                    vm.oauthCodeText = ""
                }
                .controlSize(.small)
            }
        }

        if vm.cliLoggedIn {
            HStack(spacing: 8) {
                Picker(L("机型", "GPU"), selection: $vm.wakeGPUChoice) {
                    Text("G4 · PRO 6000").tag("G4")
                    Text("H100 · 80GB").tag("H100")
                    Text("A100 · 80GB").tag("A100")
                    Text("L4 · 24GB").tag("L4")
                    Text("T4 · 16GB").tag("T4")
                }
                .labelsHidden()
                .frame(width: 138)
                .help(L(
                    "默认推荐 G4 (NVIDIA RTX PRO 6000 Blackwell · 96GB 显存)",
                    "Recommended: G4 (NVIDIA RTX PRO 6000 Blackwell · 96GB VRAM)"
                ))

                Button {
                    Task { await vm.wakeColabHeadless() }
                } label: {
                    HStack(spacing: 5) {
                        if vm.isWakingColab {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "bolt.fill")
                                .font(.caption)
                        }
                        Text(vm.isWakingColab
                             ? L("唤醒部署中...", "Deploying...")
                             : L("唤醒实例并部署", "Wake & Deploy"))
                            .fontWeight(.medium)
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.isWakingColab)
            }
        }

        HStack(spacing: 6) {
            Group {
                if vm.maskServerURL {
                    SecureField(
                        L("服务地址", "Server URL"),
                        text: $vm.colabServerURL,
                        prompt: Text(L("已开启打码保护", "URL Redacted"))
                    )
                } else {
                    TextField(
                        L("服务地址", "Server URL"),
                        text: $vm.colabServerURL,
                        prompt: Text("https://xxxx.gradio.live")
                    )
                }
            }
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 12, design: .monospaced))

            Button {
                vm.maskServerURL.toggle()
            } label: {
                Image(systemName: vm.maskServerURL ? "eye.slash.fill" : "eye")
                    .foregroundStyle(vm.maskServerURL ? Color.accentColor : Color.secondary)
            }
            .buttonStyle(.borderless)
            .help(vm.maskServerURL
                  ? L("取消打码，显示真实隧道地址", "Reveal tunnel URL")
                  : L("录屏/截图防泄密：打码隐藏公网隧道地址", "Privacy mode: redact public tunnel URL for screenshots"))

            Button(vm.isConnected ? L("刷新", "Refresh") : L("连接", "Connect")) {
                Task { await vm.checkColabStatus() }
            }
            .controlSize(.small)
            Button {
                vm.openOnboardingGuide(step: 0)
            } label: {
                Image(systemName: "questionmark.circle")
            }
            .buttonStyle(.borderless)
            .help(L("打开新手引导与部署指南", "Open quick start & deployment guide"))
        }

        HStack {
            Text(L("用户名：easycolab", "User: easycolab"))
            SecureField(L("管理密码", "Management password"), text: $vm.adminPassword)
            Button(L("复制密码", "Copy password")) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(vm.adminPassword, forType: .string)
            }
        }

        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                StatusDot(color: vm.isConnected ? .green : .orange)
                Text(vm.displayGpuName)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                Spacer(minLength: 4)
                if vm.drivePersisted {
                    Text(L("云盘持久化", "Drive Persisted"))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1.5)
                        .background(Capsule().fill(Color.primary.opacity(0.055)))
                }
                if vm.vramTotalGB > 0 || vm.ramTotalGB > 0 || vm.diskTotalGB > 0 {
                    Button {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            vm.combinedTelemetryChartMode.toggle()
                        }
                    } label: {
                        HStack(spacing: 3) {
                            Image(systemName: vm.combinedTelemetryChartMode ? "rectangle.split.1x2" : "chart.xyaxis.line")
                                .font(.system(size: 9.5, weight: .semibold))
                            Text(vm.combinedTelemetryChartMode ? L("分项", "Split") : L("同屏", "Combined"))
                                .font(.system(size: 10, weight: .medium))
                        }
                        .foregroundStyle(Color.accentColor)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            Capsule()
                                .fill(Color.accentColor.opacity(0.11))
                        )
                    }
                    .buttonStyle(.plain)
                    .help(vm.combinedTelemetryChartMode
                          ? L("切换为显存/内存/硬盘独立折线面积图", "Switch to separate VRAM / RAM / Disk area charts")
                          : L("切换为显存/内存/硬盘三线同屏对比图", "Switch to combined 3-line telemetry comparison chart"))
                }
            }

            if vm.vramTotalGB > 0 || vm.ramTotalGB > 0 || vm.diskTotalGB > 0 {
                if vm.combinedTelemetryChartMode {
                    CombinedMultiMetricChartCard(
                        vramUsed: vm.vramUsedGB,
                        vramTotal: vm.vramTotalGB,
                        vramSamples: vm.vramHistoryPct,
                        ramUsed: vm.ramUsedGB,
                        ramTotal: vm.ramTotalGB,
                        ramSamples: vm.ramHistoryPct,
                        diskUsed: vm.diskUsedGB,
                        diskTotal: vm.diskTotalGB,
                        diskSamples: vm.diskHistoryPct
                    )
                    .help(L("三线实时对比：显存 VRAM / 系统内存 RAM / 硬盘 Disk", "Live 3-metric comparison: VRAM / System RAM / Disk"))
                } else {
                    VStack(spacing: 5.5) {
                        if vm.vramTotalGB > 0 {
                            SingleMetricLineChartCard(
                                title: L("显存 VRAM", "VRAM"),
                                usedGB: vm.vramUsedGB,
                                freeGB: vm.vramFreeGB,
                                totalGB: vm.vramTotalGB,
                                samples: vm.vramHistoryPct,
                                accentColor: Color(red: 0.33, green: 0.53, blue: 0.98),
                                isWarning: vm.vramFreeGB < 8.0,
                                isCritical: vm.vramFreeGB < 3.0
                            )
                            .help(L("悬停查看历史显存占用曲线", "Hover to inspect historical GPU VRAM usage"))
                        }
                        if vm.ramTotalGB > 0 {
                            SingleMetricLineChartCard(
                                title: L("内存 RAM", "System RAM"),
                                usedGB: vm.ramUsedGB,
                                freeGB: vm.ramFreeGB,
                                totalGB: vm.ramTotalGB,
                                samples: vm.ramHistoryPct,
                                accentColor: Color(red: 0.10, green: 0.76, blue: 0.53),
                                isWarning: vm.ramFreeGB < 10.0,
                                isCritical: vm.ramFreeGB < 5.0
                            )
                            .help(L("悬停查看系统内存与热备占用曲线", "Hover to inspect system RAM & hot-standby usage"))
                        }
                        if vm.diskTotalGB > 0 {
                            SingleMetricLineChartCard(
                                title: L("硬盘 Disk", "Disk"),
                                usedGB: vm.diskUsedGB,
                                freeGB: vm.diskFreeGB,
                                totalGB: vm.diskTotalGB,
                                samples: vm.diskHistoryPct,
                                accentColor: Color(red: 0.05, green: 0.73, blue: 0.88),
                                isWarning: vm.diskFreeGB < 20.0,
                                isCritical: vm.diskFreeGB < 10.0
                            )
                            .help(L("悬停查看本地实例 NVMe 磁盘空间曲线", "Hover to inspect instance NVMe disk usage"))
                        }
                    }
                }

                if vm.shmTotalGB >= 1.0 || vm.driveTotalGB > 0 {
                    HStack(spacing: 6) {
                        if vm.shmTotalGB >= 1.0 {
                            Text(L(
                                "内存盘可用 \(fmt(vm.shmFreeGB, 1))/\(fmt(vm.shmTotalGB, 0))G",
                                "/dev/shm Free \(fmt(vm.shmFreeGB, 1))/\(fmt(vm.shmTotalGB, 0))G"
                            ))
                        }
                        Spacer()
                        if vm.driveTotalGB > 0 {
                            Text(L(
                                "云盘可用 \(fmt(vm.driveFreeGB, 1))G",
                                "Drive Free \(fmt(vm.driveFreeGB, 1))G"
                            ))
                        }
                    }
                    .font(.system(size: 10.5))
                    .foregroundStyle(.secondary.opacity(0.85))
                    .monospacedDigit()
                    .padding(.horizontal, 2)
                }
            }
        }
        .padding(.top, 2)
    }

    @ViewBuilder
    var usageRows: some View {
        HStack(spacing: 0) {
            metricColumn(
                title: vm.officialCUBalance != nil ? L("账户余额", "CU Balance") : L("预计余额", "Est. Balance"),
                value: "\(fmt(vm.displayRemainingCU)) CU",
                highlight: .primary
            )
            Divider().frame(height: 28)
            metricColumn(
                title: L("扣费速率", "Burn Rate"),
                value: "\(fmt(vm.displayBurnRate)) CU/h",
                highlight: vm.displayBurnRate > 0 ? .orange : .secondary
            )
            Divider().frame(height: 28)
            metricColumn(
                title: L("本次运行", "Uptime"),
                value: vm.formattedUptime,
                highlight: .secondary
            )
        }
        .padding(.vertical, 4)

        HStack {
            Text(L("闲置自动释放", "Idle Auto-Sleep"))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Picker(L("闲置自动释放", "Idle Auto-Sleep"), selection: Binding(
                get: { vm.idleTimeoutMinutes },
                set: { newVal in Task { await vm.updateIdleTimeoutConfig(newVal) } }
            )) {
                Text(L("15 分钟", "15m")).tag(15)
                Text(L("30 分钟", "30m")).tag(30)
                Text(L("60 分钟", "60m")).tag(60)
                Text(L("关闭", "Off")).tag(0)
            }
            .labelsHidden()
            .controlSize(.small)
            .frame(width: 105)
        }

        if vm.isConnected && vm.idleTimeoutMinutes > 0 {
            Text(vm.formattedIdleRemaining)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }

        Button(role: .destructive) {
            vm.showDisconnectConfirm = true
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "power")
                    .font(.caption)
                Text(vm.activeVMCount > 0
                     ? L("释放全部实例 (\(vm.activeVMCount) 台运行中)", "Release All Instances (\(vm.activeVMCount) Running)")
                     : L("释放全部实例", "Release All Instances"))
            }
            .frame(maxWidth: .infinity)
        }
        .controlSize(.small)
    }

    func metricColumn(title: String, value: String, highlight: Color) -> some View {
        VStack(spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(highlight)
                .monospacedDigit()
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    func openModelHubRows(palette: StudioPalette) -> some View {
        VStack(spacing: 5) {
            ForEach(vm.allAvailableModels) { spec in
                let isSelected = vm.selectedModelId == spec.id
                let isLoaded = vm.loadedModelIds.contains(spec.id) || (vm.loadedModelId == spec.id && vm.modelLoaded)
                Button {
                    withAnimation(.easeInOut(duration: 0.16)) {
                        vm.switchSelectedModel(spec)
                    }
                } label: {
                    HStack(spacing: 9) {
                        Image(systemName: spec.icon)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(isSelected ? .white : Color.accentColor)
                            .frame(width: 26, height: 26)
                            .background(
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(isSelected ? Color.accentColor : Color.accentColor.opacity(0.12))
                            )

                        VStack(alignment: .leading, spacing: 1.5) {
                            HStack(spacing: 5) {
                                Text(spec.name)
                                    .font(.system(size: 12.5, weight: .semibold))
                                    .foregroundStyle(.primary)
                                    .lineLimit(1)
                                if spec.modality == .comfyUI && vm.comfyRunning {
                                    Text(vm.comfyQueueRunning > 0 ? L("渲染中", "Rendering") : L("画布在线", "Canvas Live"))
                                        .font(.system(size: 9.5, weight: .bold))
                                        .foregroundStyle(Color.green)
                                        .padding(.horizontal, 5)
                                        .padding(.vertical, 1)
                                        .background(Capsule().fill(Color.green.opacity(0.15)))
                                } else if isLoaded {
                                    Text(L("显存常驻", "In VRAM"))
                                        .font(.system(size: 9.5, weight: .bold))
                                        .foregroundStyle(Color.green)
                                        .padding(.horizontal, 5)
                                        .padding(.vertical, 1)
                                        .background(Capsule().fill(Color.green.opacity(0.15)))
                                }
                            }
                            Text(spec.badge)
                                .font(.system(size: 10.5))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        Spacer(minLength: 4)
                        if spec.isCustom {
                            Button {
                                Task { await vm.removeCustomHFModel(spec) }
                            } label: {
                                Image(systemName: "trash")
                                    .font(.system(size: 10.5))
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                            .help(L("移除自定义模型 \(spec.name)", "Remove custom model \(spec.name)"))
                        }
                        if isSelected {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 13))
                                .foregroundStyle(Color.accentColor)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(isSelected ? Color.accentColor.opacity(0.11) : palette.inputFill.opacity(0.6))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(isSelected ? Color.accentColor.opacity(0.55) : Color.clear, lineWidth: 1)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .contextMenu {
                    if spec.isCustom {
                        Button(role: .destructive) {
                            Task { await vm.removeCustomHFModel(spec) }
                        } label: {
                            Label(L("移除该自定义模型", "Remove Custom Model"), systemImage: "trash")
                        }
                    }
                }
            }

            Button {
                vm.customModelErrorMessage = ""
                vm.showAddCustomModelSheet = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "plus.circle.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Color.accentColor)
                    Text(L("添加自定义 HuggingFace 模型...", "Add Custom HuggingFace Repo..."))
                        .font(.system(size: 11.5, weight: .semibold))
                        .foregroundStyle(Color.accentColor)
                    Spacer()
                    Text("GGUF / Safetensors")
                        .font(.system(size: 9.5, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.accentColor.opacity(0.08))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(Color.accentColor.opacity(0.35), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                )
            }
            .buttonStyle(.plain)
        }

        if !vm.loadedModelsDetail.isEmpty {
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Image(systemName: "memorychip.fill")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Color.green)
                    Text(L(
                        "GPU 显存共驻模型 (\(vm.loadedModelsDetail.count))",
                        "Co-Resident in VRAM (\(vm.loadedModelsDetail.count))"
                    ))
                        .font(.system(size: 10.5, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if vm.loadedModelsDetail.count > 1 {
                        Button(L("全部释放", "Unload All")) {
                            Task { await vm.unloadRemoteModelFromVRAM(modelId: "all") }
                        }
                        .buttonStyle(.borderless)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.red)
                        .disabled(vm.isUnloadingModel)
                    }
                }
                ForEach(vm.loadedModelsDetail) { item in
                    HStack(spacing: 6) {
                        StatusDot(color: .green)
                        Text(item.localizedName)
                            .font(.system(size: 11, weight: .medium))
                            .lineLimit(1)
                        Spacer(minLength: 4)
                        if let ctx = item.context_length, ctx > 0 {
                            Text("\(ctx / 1024)K")
                                .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                                .foregroundStyle(Color.accentColor)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Capsule().fill(Color.accentColor.opacity(0.12)))
                        }
                        if let gb = item.vram_gb, gb > 0 {
                            Text("~\(fmt(gb, 1))G")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            Task { await vm.unloadRemoteModelFromVRAM(modelId: item.id) }
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 12))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help(L("从 GPU 显存卸载 \(item.localizedName)", "Unload \(item.localizedName) from GPU VRAM"))
                        .disabled(vm.isUnloadingModel)
                    }
                    .padding(.horizontal, 7)
                    .padding(.vertical, 4)
                    .background(RoundedRectangle(cornerRadius: 6).fill(palette.inputFill))
                }
            }
            .padding(.top, 2)
        }

        Divider().padding(.vertical, 2)

        if vm.activeModelSpec.modality == .comfyUI {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(L("画布状态", "Canvas Status"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(vm.comfyRunning
                         ? L("在线 (端口 8188)", "Online (Port 8188)")
                         : (vm.comfyStarting ? L("部署启动中...", "Starting...") : L("未启动", "Stopped")))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(vm.comfyRunning ? Color.green : (vm.comfyStarting ? Color.orange : Color.secondary))
                }
                if vm.comfyStarting || vm.comfyDownloadingWeight {
                    ProgressView(value: vm.comfyProgress, total: 1.0)
                }
                HStack(spacing: 6) {
                    if vm.comfyRunning && !vm.comfyPublicURL.isEmpty {
                        Button {
                            vm.openComfyUIInBrowser()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "safari")
                                Text(L("打开画布", "Open Canvas"))
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.regular)

                        Button {
                            Task { await vm.stopRemoteComfyUI() }
                        } label: {
                            Image(systemName: "stop.fill")
                        }
                        .controlSize(.regular)
                        .help(L("同步云盘并停止 ComfyUI 进程", "Sync to Google Drive & stop ComfyUI process"))
                    } else {
                        Button {
                            Task { await vm.startRemoteComfyUI(forceRestart: false, autoOpenBrowser: true) }
                        } label: {
                            HStack(spacing: 4) {
                                if vm.comfyStarting {
                                    ProgressView().controlSize(.small)
                                } else {
                                    Image(systemName: "play.circle.fill")
                                }
                                Text(vm.comfyStarting ? L("正在部署...", "Deploying...") : L("启动 ComfyUI 画布", "Start ComfyUI Canvas"))
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.regular)
                        .disabled(!vm.isConnected || vm.comfyStarting)
                    }
                }
            }
        } else {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(L("HF 仓库", "HF Repo"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        if let hfURL = URL(string: "https://huggingface.co/\(vm.activeModelSpec.repoId)") {
                            NSWorkspace.shared.open(hfURL)
                        }
                    } label: {
                        HStack(spacing: 3) {
                            Text(vm.activeModelSpec.repoId)
                                .font(.system(size: 10.5, design: .monospaced))
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Image(systemName: "arrow.up.right.square")
                                .font(.system(size: 9.5))
                        }
                        .foregroundStyle(Color.accentColor)
                    }
                    .buttonStyle(.plain)
                    .help(L("在浏览器打开 HuggingFace 模型仓库", "Open HuggingFace model repository in browser"))
                }

                HStack {
                    Text(L("磁盘权重", "Disk Weights"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(vm.allDownloaded
                         ? L("已就绪 \(vm.downloadedCount)/\(vm.totalFileCount) (\(fmt(vm.totalDiskSizeGB, 1)) GB)",
                             "Ready \(vm.downloadedCount)/\(vm.totalFileCount) (\(fmt(vm.totalDiskSizeGB, 1)) GB)")
                         : L("\(vm.downloadedCount)/\(vm.totalFileCount) 个文件",
                             "\(vm.downloadedCount)/\(vm.totalFileCount) files"))
                        .font(.caption.weight(.medium))
                        .foregroundStyle(vm.allDownloaded ? Color.green : Color.secondary)
                        .monospacedDigit()
                }

                DisclosureGroup(
                    L("查看权重分片 (\(vm.modelFiles.count))", "View Weight Shards (\(vm.modelFiles.count))"),
                    isExpanded: $showFileList
                ) {
                    VStack(alignment: .leading, spacing: 6) {
                        ScrollView {
                            VStack(spacing: 4) {
                                ForEach(vm.modelFiles) { file in
                                    HStack(spacing: 5) {
                                        Image(systemName: file.exists ? "checkmark.circle.fill" : "circle.dashed")
                                            .foregroundStyle(file.exists ? Color.green : Color.secondary)
                                            .font(.system(size: 10))
                                        VStack(alignment: .leading, spacing: 1) {
                                            Text(file.name)
                                                .font(.system(size: 10.5, design: .monospaced))
                                                .lineLimit(1)
                                                .truncationMode(.middle)
                                            if let desc = file.localizedDesc, !desc.isEmpty {
                                                Text(desc)
                                                    .font(.system(size: 9.5))
                                                    .foregroundStyle(.secondary)
                                                    .lineLimit(1)
                                            }
                                        }
                                        Spacer()
                                        Text(file.exists ? "\(fmt(file.size_gb, 1))G" : L("未下", "Pending"))
                                            .font(.system(size: 10))
                                            .foregroundStyle(.secondary)
                                            .monospacedDigit()
                                    }
                                    .help(file.localizedDesc ?? file.name)
                                }
                            }
                            .padding(.top, 4)
                        }
                        .frame(maxHeight: 125)

                        HStack(spacing: 6) {
                            Text("HF Token")
                                .font(.system(size: 10.5))
                                .foregroundStyle(.secondary)
                            SecureField(
                                "HF Token",
                                text: $vm.hfToken,
                                prompt: Text(L("可选，加速下载或访问受限仓库", "Optional, faster / gated repos"))
                            )
                            .textFieldStyle(.roundedBorder)
                            .controlSize(.small)
                            .font(.system(size: 11, design: .monospaced))
                        }
                    }
                }
                .font(.caption)

                if vm.activeModelSpec.modality == .llmVLM {
                    let presetKs = [8, 16, 32, 64, 128, 192, 256]
                    let pickerKs = presetKs.contains(vm.contextLengthK)
                        ? presetKs
                        : (presetKs + [vm.contextLengthK]).sorted()

                    HStack {
                        Text(L("上下文长度", "Context Length"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Picker(L("上下文长度", "Context Length"), selection: Binding(
                            get: { vm.contextLengthK },
                            set: { newK in
                                vm.contextLengthK = newK.clamped(to: 4...256)
                                Task { await vm.syncContextLengthToColab(announce: true) }
                            }
                        )) {
                            ForEach(pickerKs, id: \.self) { k in
                                Text(k == 256 ? L("256K (最大)", "256K (Max)") : "\(k)K (\(k * 1024))").tag(k)
                            }
                        }
                        .labelsHidden()
                        .controlSize(.small)
                        .frame(width: 148)
                    }

                    if vm.selectedModelId == "qwen38_27b_gguf",
                       vm.isSelectedModelLoadedInVRAM,
                       vm.loadedContextLengthTokens > 0,
                       vm.loadedContextLengthTokens != vm.contextLengthTokens {
                        Text(L(
                            "当前显存 KV 为 \(vm.loadedContextLengthTokens / 1024)K，点「重新装载」切换至 \(vm.contextLengthK)K",
                            "VRAM KV cache is \(vm.loadedContextLengthTokens / 1024)K; click 'Reload' to apply \(vm.contextLengthK)K"
                        ))
                            .font(.system(size: 10.5))
                            .foregroundStyle(.orange)
                    } else if vm.selectedModelId == "qwen38_27b_gguf", vm.contextLengthK >= 128 {
                        Text(L(
                            "提示：日常对话选 32K/64K 可达峰值 90~110 tok/s；256K 预分配超长槽位约为 80+ tok/s",
                            "Tip: 32K/64K hits peak 90–110 tok/s; 256K pre-allocated context runs at ~80+ tok/s"
                        ))
                            .font(.system(size: 10.5))
                            .foregroundStyle(.secondary)
                    }
                }

                if !vm.isConnected {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(.orange)
                        Text(L("Colab 实例离线（已闲置释放或未连接）", "Colab offline (idle-released or disconnected)"))
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.orange)
                            .lineLimit(1)
                        Spacer(minLength: 4)
                        if vm.cliLoggedIn {
                            Button(vm.isWakingColab ? L("唤醒中...", "Waking...") : L("一键唤醒", "Wake")) {
                                Task { await vm.wakeColabHeadless() }
                            }
                            .controlSize(.mini)
                            .buttonStyle(.borderedProminent)
                            .disabled(vm.isWakingColab)
                        } else if !vm.normalizedBaseURL.isEmpty {
                            Button(L("重连", "Reconnect")) {
                                Task { await vm.checkColabStatus() }
                            }
                            .controlSize(.mini)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .background(RoundedRectangle(cornerRadius: 7).fill(Color.orange.opacity(0.10)))
                }

                if vm.isDownloadingWeights || vm.isLoadingModel {
                    ProgressView(value: vm.modelTaskProgress, total: 1.0)
                }

                HStack(spacing: 6) {
                    Button {
                        Task { await vm.startRemoteDownloadWeights() }
                    } label: {
                        HStack(spacing: 4) {
                            if vm.isDownloadingWeights {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "arrow.down.circle")
                            }
                            Text(vm.isDownloadingWeights
                                 ? L("下载中...", "Downloading...")
                                 : (vm.allDownloaded ? L("校验权重", "Verify Weights") : L("下载权重", "Download Weights")))
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .controlSize(.regular)
                    .disabled(!vm.isConnected || vm.isDownloadingWeights || vm.isLoadingModel || vm.isUnloadingModel)

                    Button {
                        Task { await vm.loadRemoteModelToVRAM() }
                    } label: {
                        HStack(spacing: 4) {
                            if vm.isLoadingModel {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "memorychip")
                            }
                            Text(vm.isLoadingModel
                                 ? L("装载中...", "Loading...")
                                 : (vm.isSelectedModelLoadedInVRAM ? L("重新装载", "Reload") : L("加载到显存", "Load to VRAM")))
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.regular)
                    .disabled(!vm.isConnected || vm.isDownloadingWeights || vm.isLoadingModel || vm.isUnloadingModel)

                    if vm.isSelectedModelLoadedInVRAM {
                        Button {
                            Task { await vm.unloadRemoteModelFromVRAM(modelId: vm.selectedModelId) }
                        } label: {
                            Image(systemName: "eject.fill")
                        }
                        .controlSize(.regular)
                        .help(L("将当前模型从 GPU 显存中卸载释放", "Unload active model from GPU VRAM"))
                        .disabled(vm.isUnloadingModel)
                    }
                }
            }
        }
    }

    @ViewBuilder
    var adaptiveParamRows: some View {
        switch vm.activeModelSpec.modality {
        case .comfyUI:
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(L("持久化架构", "Persistence"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(vm.displayComfyPersistenceLabel)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(vm.comfyDrivePersisted ? Color.green : Color.secondary)
                }

                HStack {
                    Text(L("已存节点 / 权重", "Saved Nodes / Weights"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(L(
                        "\(vm.comfyCustomNodesCount) 个节点 · \(vm.comfyModelsCount) 个权重",
                        "\(vm.comfyCustomNodesCount) nodes · \(vm.comfyModelsCount) weights"
                    ))
                        .font(.caption.weight(.medium))
                        .monospacedDigit()
                }

                HStack {
                    Text(L("画布执行队列", "Execution Queue"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(L(
                        "运行 \(vm.comfyQueueRunning) · 排队 \(vm.comfyQueuePending)",
                        "Running \(vm.comfyQueueRunning) · Pending \(vm.comfyQueuePending)"
                    ))
                        .font(.caption.weight(.medium))
                        .foregroundStyle(vm.comfyQueueRunning > 0 ? Color.green : .secondary)
                        .monospacedDigit()
                }

                Toggle(L("启动前释放其它常驻模型显存", "Unload other VRAM models before starting"), isOn: $vm.comfyUnloadOthersOnStart)
                    .font(.caption)

                HStack(spacing: 6) {
                    Button {
                        Task { await vm.syncRemoteComfyUIToDrive() }
                    } label: {
                        HStack(spacing: 4) {
                            if vm.comfySyncingDrive {
                                ProgressView().controlSize(.mini)
                            } else {
                                Image(systemName: "arrow.triangle.2.circlepath.icloud")
                            }
                            Text(vm.comfySyncingDrive ? L("同步中...", "Syncing...") : L("立即同步云盘", "Sync to Drive"))
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .controlSize(.small)
                    .disabled(!vm.isConnected || vm.comfySyncingDrive)

                    Button {
                        Task { await vm.freeRemoteComfyUIVRAM() }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "memorychip")
                            Text(L("清缓存显存", "Free VRAM Cache"))
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .controlSize(.small)
                    .disabled(!vm.isConnected || !vm.comfyRunning)
                }
            }

        case .llmVLM:
            Toggle(L("启用深度思考链 (Thinking Mode)", "Enable Thinking Chain (Thinking Mode)"), isOn: $vm.enableThinking)
                .font(.callout)

            if vm.enableThinking {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(L("思考深度 (Reasoning Effort)", "Reasoning Effort"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(vm.reasoningEffort.uppercased())
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundStyle(Color.accentColor)
                    }
                    HStack(spacing: 4) {
                        ForEach([
                            ("low", L("精简", "Low")),
                            ("medium", L("均衡", "Med")),
                            ("high", L("深入", "High")),
                            ("xhigh", L("极深", "XHigh"))
                        ], id: \.0) { (key, title) in
                            let active = vm.reasoningEffort == key
                            Button {
                                vm.reasoningEffort = key
                            } label: {
                                Text(title)
                                    .font(.system(size: 10.5, weight: active ? .bold : .medium))
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 3.5)
                                    .background(
                                        RoundedRectangle(cornerRadius: 5)
                                            .fill(active ? Color.accentColor.opacity(0.18) : Color.primary.opacity(0.05))
                                    )
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 5)
                                            .strokeBorder(active ? Color.accentColor.opacity(0.65) : Color.clear, lineWidth: 1)
                                    )
                                    .foregroundStyle(active ? Color.accentColor : Color.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 5) {
                    Text(L("上下文窗口 (最高 256K)", "Context Window (Up to 256K)"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    TextField(
                        "K",
                        value: Binding(
                            get: { vm.contextLengthK },
                            set: { vm.contextLengthK = $0.clamped(to: 4...256) }
                        ),
                        format: .number
                    )
                    .textFieldStyle(.roundedBorder)
                    .controlSize(.small)
                    .font(.system(size: 11.5, weight: .semibold, design: .monospaced))
                    .multilineTextAlignment(.trailing)
                    .frame(width: 50)
                    .onSubmit {
                        Task { await vm.syncContextLengthToColab(announce: true) }
                    }
                    Text("K")
                        .font(.caption.weight(.semibold))
                    Stepper(
                        "",
                        value: Binding(
                            get: { vm.contextLengthK },
                            set: { newK in
                                vm.contextLengthK = newK.clamped(to: 4...256)
                                Task { await vm.syncContextLengthToColab(announce: true) }
                            }
                        ),
                        in: 4...256,
                        step: 4
                    )
                    .labelsHidden()
                    .controlSize(.small)
                }

                Slider(
                    value: Binding(
                        get: { Double(vm.contextLengthK) },
                        set: { vm.contextLengthK = Int(($0 / 4.0).rounded() * 4.0).clamped(to: 4...256) }
                    ),
                    in: 4...256,
                    onEditingChanged: { editing in
                        if !editing {
                            Task { await vm.syncContextLengthToColab(announce: true) }
                        }
                    }
                )

                HStack(spacing: 4) {
                    ForEach([8, 16, 32, 64, 128, 256], id: \.self) { k in
                        let active = vm.contextLengthK == k
                        Button {
                            vm.contextLengthK = k
                            Task { await vm.syncContextLengthToColab(announce: true) }
                        } label: {
                            Text("\(k)K")
                                .font(.system(size: 10, weight: active ? .bold : .medium, design: .monospaced))
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 3)
                                .background(
                                    RoundedRectangle(cornerRadius: 5)
                                        .fill(active ? Color.accentColor.opacity(0.18) : Color.primary.opacity(0.05))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 5)
                                        .strokeBorder(active ? Color.accentColor.opacity(0.65) : Color.clear, lineWidth: 1)
                                )
                                .foregroundStyle(active ? Color.accentColor : Color.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }

                Text(L(
                    "总容量 \(vm.contextLengthTokens) Tokens" + (vm.loadedContextLengthTokens > 0 ? " · 当前已加载 \(vm.loadedContextLengthTokens / 1024)K" : ""),
                    "Capacity \(vm.contextLengthTokens) tokens" + (vm.loadedContextLengthTokens > 0 ? " · Loaded \(vm.loadedContextLengthTokens / 1024)K" : "")
                ))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(L("最大输出长度 (含思考链)", "Max Output Tokens (incl. Thinking)"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(Int(vm.maxNewTokens)) Tokens")
                        .font(.caption.weight(.medium))
                        .monospacedDigit()
                }
                Slider(
                    value: Binding(
                        get: { vm.maxNewTokens },
                        set: { vm.maxNewTokens = (($0 / 512).rounded() * 512).clamped(to: 1024...65536) }
                    ),
                    in: 1024...65536
                )
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(L("采样温度 (Temperature)", "Temperature"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(fmt(vm.temperature, 2)) · Top-P \(fmt(vm.topP, 2)) · Top-K 20")
                        .font(.caption.weight(.medium))
                        .monospacedDigit()
                }
                Slider(value: $vm.temperature, in: 0.1...1.5)
            }

            Button {
                vm.chatHistory.removeAll()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "trash")
                        .font(.caption)
                    Text(L("清空多轮对话历史", "Clear Chat History"))
                }
                .frame(maxWidth: .infinity)
            }
            .controlSize(.small)
            .disabled(vm.chatHistory.isEmpty || vm.isGenerating)

        case .image:
            VStack(alignment: .leading, spacing: 4) {
                Text(L("输出画幅", "Output Aspect & Resolution"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Picker(L("画幅", "Resolution"), selection: $vm.selectedImageResolution) {
                    ForEach(vm.activeImageResolutions) { opt in
                        Text(opt.label).tag(opt)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: .infinity)
            }

            if vm.isMiniMaxH3ImageModel {
                HStack {
                    Text(L("采样步数", "Sampling Steps"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(L("6 步 (PDD 蒸馏固定)", "6 steps (PDD Distilled)"))
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.primary)
                        .monospacedDigit()
                }
                .padding(.vertical, 2)
            } else {
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(L("采样步数", "Sampling Steps"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(L("\(Int(vm.inferenceSteps)) 步", "\(Int(vm.inferenceSteps)) steps"))
                            .font(.caption.weight(.medium))
                            .monospacedDigit()
                    }
                    Slider(
                        value: Binding(
                            get: { vm.inferenceSteps },
                            set: { vm.inferenceSteps = $0.rounded().clamped(to: 4...50) }
                        ),
                        in: 4...50
                    )
                }
            }

            if vm.selectedModelId == "z_image_turbo" && vm.taskModeId == "i2i_edit" {
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(L("重绘幅度", "Denoising Strength"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(fmt(vm.denoisingStrength, 2))
                            .font(.caption.weight(.medium))
                            .monospacedDigit()
                    }
                    Slider(value: $vm.denoisingStrength, in: 0.20...1.0)
                }
            }

            seedAndBatchControls(unitLabelZh: "张")

        case .videoAudio, .video:
            VStack(alignment: .leading, spacing: 4) {
                Text(L("视频分辨率", "Video Resolution"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Picker(L("分辨率", "Resolution"), selection: $vm.selectedVideoResolution) {
                    ForEach(VIDEO_RESOLUTIONS) { opt in
                        Text(opt.label).tag(opt)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: .infinity)
            }

            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text(L("视频总帧数 (4~15 秒)", "Total Frames (4–15s)"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(L(
                        "\(Int(vm.numFrames)) 帧 · 约 \(fmt(vm.numFrames / vm.fps, 1)) 秒",
                        "\(Int(vm.numFrames)) frames · ~\(fmt(vm.numFrames / vm.fps, 1))s"
                    ))
                        .font(.caption.weight(.medium))
                        .monospacedDigit()
                }
                Slider(
                    value: Binding(
                        get: { vm.numFrames },
                        set: { vm.numFrames = (((($0 - 5) / 17).rounded() * 17) + 5).clamped(to: 90...362) }
                    ),
                    in: 90...362
                )
                HStack(spacing: 4) {
                    ForEach([(90, "4s"), (124, "5s"), (192, "8s"), (243, "10s"), (294, "12s"), (362, "15s")], id: \.0) { frames, tag in
                        let isSel = Int(vm.numFrames) == frames
                        Button {
                            vm.numFrames = Double(frames)
                        } label: {
                            Text(tag)
                                .font(.system(size: 10, weight: isSel ? .semibold : .regular, design: .monospaced))
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 2.5)
                                .background(
                                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                                        .fill(isSel ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.10))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                                        .stroke(isSel ? Color.accentColor.opacity(0.55) : Color.clear, lineWidth: 1)
                                )
                                .foregroundStyle(isSel ? Color.accentColor : Color.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(L("采样步数", "Sampling Steps"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(L("\(Int(vm.inferenceSteps)) 步", "\(Int(vm.inferenceSteps)) steps"))
                        .font(.caption.weight(.medium))
                        .monospacedDigit()
                }
                Slider(
                    value: Binding(
                        get: { vm.inferenceSteps },
                        set: { vm.inferenceSteps = (($0 / 5).rounded() * 5).clamped(to: 10...60) }
                    ),
                    in: 10...60
                )
            }

            seedAndBatchControls(unitLabelZh: "条")
        }
    }

    @ViewBuilder
    func seedAndBatchControls(unitLabelZh: String) -> some View {
        HStack {
            Text(L("随机种子", "Seed"))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            TextField(L("种子", "Seed"), value: $vm.seed, format: .number.grouping(.never))
                .textFieldStyle(.roundedBorder)
                .controlSize(.small)
                .frame(width: 96)
                .disabled(vm.randomizeSeed)
            Toggle(L("随机", "Random"), isOn: $vm.randomizeSeed)
                .toggleStyle(.checkbox)
                .font(.caption)
        }

        HStack(spacing: 8) {
            Text(L("批量连抽", "Batch Size"))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            HStack(spacing: 2) {
                ForEach([1, 2, 3, 4], id: \.self) { count in
                    let isSelected = vm.batchCount == count
                    Button {
                        withAnimation(.easeInOut(duration: 0.14)) {
                            vm.batchCount = count
                        }
                    } label: {
                        Text(L("\(count) \(unitLabelZh)", "\(count)×"))
                            .font(.system(size: 11.5, weight: isSelected ? .semibold : .regular))
                            .foregroundStyle(isSelected ? Color.white : Color.primary)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 4)
                            .background(
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(isSelected ? Color.accentColor : Color.clear)
                            )
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(2)
            .frame(width: 192)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.primary.opacity(colorScheme == .dark ? 0.12 : 0.06))
            )
        }
    }


    // MARK: - 右侧独立全高运行日志面板

    func logLineColor(_ line: String) -> Color {
        let lower = line.lowercased()
        if line.contains("失败") || line.contains("错误") || line.contains("异常")
            || line.contains("ERROR") || lower.contains("failed") || lower.contains("error:")
            || lower.contains("exception") || lower.contains("traceback") {
            return Color(red: 0.98, green: 0.46, blue: 0.46)
        }
        if line.contains("───") {
            return Color(red: 0.46, green: 0.80, blue: 0.98)
        }
        if line.contains("%") || line.contains("已装载") || line.contains("成功") || line.contains("完成")
            || lower.contains("loaded") || lower.contains("succeeded") || lower.contains("successfully")
            || lower.contains("completed") || lower.contains("ready in vram") {
            return Color(red: 0.35, green: 0.92, blue: 0.55)
        }
        return Color(white: 0.87)
    }

    func rightLogPanel(palette: StudioPalette) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            // 实时任务摘要横条
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: isTaskStreaming ? "bolt.horizontal.circle.fill" : "info.circle")
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(isTaskStreaming ? Color.accentColor : Color.secondary)
                    .padding(.top, 1)

                VStack(alignment: .leading, spacing: 3) {
                    Text(vm.displayStatusMessage)
                        .font(.system(size: 11.5, weight: .medium))
                        .foregroundStyle(.primary)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)

                    if vm.isDownloadingWeights || vm.isLoadingModel {
                        ProgressView(value: vm.modelTaskProgress)
                            .progressViewStyle(.linear)
                            .controlSize(.small)
                    } else if vm.isGenerating && vm.progressValue > 0 {
                        ProgressView(value: vm.progressValue)
                            .progressViewStyle(.linear)
                            .controlSize(.small)
                    }
                }

                Spacer(minLength: 0)
            }
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: 9)
                    .fill(palette.sidebarCardFill)
                    .overlay(
                        RoundedRectangle(cornerRadius: 9)
                            .strokeBorder(palette.cardBorder, lineWidth: 1)
                    )
            )

            // 全高自适应深色终端流视口
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 5) {
                        ForEach(Array(vm.liveConsoleLines.enumerated()), id: \.offset) { idx, line in
                            let renderedLine = localizeConsoleLine(line)
                            Text(renderedLine)
                                .font(.system(size: 11.5, design: .monospaced))
                                .foregroundStyle(logLineColor(renderedLine))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                                .id(idx)
                        }
                    }
                    .padding(12)
                }
                .environment(\.colorScheme, .dark)
                .compatScrollIndicatorMargins(.all, 4)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(palette.logTerminalFill)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
                        )
                )
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .onAppear {
                    if !vm.liveConsoleLines.isEmpty {
                        proxy.scrollTo(vm.liveConsoleLines.count - 1, anchor: .bottom)
                    }
                }
                .compatOnChange(of: vm.liveConsoleLines) { newLines in
                    if !newLines.isEmpty {
                        proxy.scrollTo(newLines.count - 1, anchor: .bottom)
                    }
                }
            }
        }
        .padding(12)
    }
}

