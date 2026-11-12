import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers
import ObjectiveC.runtime

// MARK: - MainStudioView 扩展：主工作区、ComfyUI 节点工作台、API 协议网关与流式对话/媒体预览

extension MainStudioView {
    // MARK: - 右侧主工作区

    func workspacePanel(palette: StudioPalette) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // 1. 顶部模型概览与醒目卡片模式切换栏
                modeSwitcherHeader(palette: palette)

                if vm.activeModelSpec.modality == .comfyUI {
                    comfyUIWorkspaceSection(palette: palette)
                } else if vm.taskModeId == "api" {
                    // API 协议网关专属面板（OpenAI + Anthropic 双协议配置与一键复制）
                    apiGatewaySection(palette: palette)
                } else {
                    // 2. 参考图上传区（在首尾帧、图生视频、指令修图、视觉理解模式下自动展开）
                    if ["fl2va", "i2v", "i2i_edit", "vlm"].contains(vm.taskModeId) {
                        referenceImageSection
                    }

                    // 3. 提示词与输入工作台
                    promptComposerSection(palette: palette)

                    // 4. 生成主按钮与状态条
                    generateActionSection

                    // 5. 自适应预览区（大模型流式对话区 或 视频/图像播放器 + 历史库）
                    if vm.activeModelSpec.modality == .llmVLM {
                        llmConversationCard(palette: palette)
                    } else {
                        HStack(alignment: .top, spacing: 14) {
                            mediaPreviewCard
                            mediaHistoryCard.frame(width: 242)
                        }
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: 1180)
            .frame(maxWidth: .infinity)
        }
        .compatScrollIndicatorMargins(.vertical, 6)
    }

    // MARK: - ComfyUI 云端节点视频工作台与 Google Drive 软链持久化面板

    @ViewBuilder
    func comfyUIWorkspaceSection(palette: StudioPalette) -> some View {
        let modelSubfolders = [
            ("diffusion_models", L("diffusion_models (DiT / 视频主干模型)", "diffusion_models (DiT / Video Backbones)")),
            ("checkpoints", L("checkpoints (完整大模型)", "checkpoints (Full Checkpoints)")),
            ("loras", L("loras (LoRA 风格/运镜微调)", "loras (Style & Camera LoRAs)")),
            ("vae", L("vae (3D 视频 / 图像 VAE)", "vae (3D Video / Image VAE)")),
            ("text_encoders", L("text_encoders (UMT5 / Qwen / CLIP 编码器)", "text_encoders (UMT5 / Qwen / CLIP)")),
            ("clip_vision", L("clip_vision (首尾帧 / 图生视频视觉编码器)", "clip_vision (I2V / FLF2V Vision Encoders)")),
            ("controlnet", L("controlnet (姿态 / 深度 / VACE 控制)", "controlnet (Pose / Depth / VACE)")),
            ("upscale_models", L("upscale_models (高清超分放大)", "upscale_models (Super-Resolution Upscalers)")),
            ("audio_encoders", L("audio_encoders (音频驱动 / 配音节点)", "audio_encoders (Audio-Driven / Foley Nodes)")),
            ("gguf", L("gguf (量化视频/图像主干)", "gguf (Quantized Video / Image Backbones)"))
        ]

        // 1. 核心画布拉起与全屏浏览器入口卡片
        WorkspaceCard(
            title: L("云端 ComfyUI 全屏节点画布", "Cloud ComfyUI Fullscreen Node Canvas"),
            subtitle: L(
                "预装 ComfyUI + ComfyUI-Manager + VideoHelperSuite，自动复用 EasyColab 的 G4 保活、显存/内存/磁盘折线监控与自动休眠",
                "Preinstalled with ComfyUI + ComfyUI-Manager + VideoHelperSuite, sharing EasyColab's G4 keep-alive, live VRAM/RAM/Disk telemetry & idle auto-sleep"
            ),
            icon: "point.3.connected.trianglepath.dotted",
            trailing: AnyView(
                HStack(spacing: 6) {
                    StatusDot(color: vm.comfyRunning ? .green : (vm.comfyStarting ? .orange : .secondary.opacity(0.45)))
                    Text(vm.comfyRunning
                         ? (vm.comfyQueueRunning > 0
                            ? L("队列渲染中 (\(vm.comfyQueueRunning) 运行 / \(vm.comfyQueuePending) 排队)",
                                "Rendering (\(vm.comfyQueueRunning) active / \(vm.comfyQueuePending) queued)")
                            : L("画布在线待命", "Canvas Ready"))
                         : (vm.comfyStarting ? L("正在部署启动...", "Starting...") : L("未启动", "Stopped")))
                        .font(.system(size: 11.5, weight: .semibold))
                        .foregroundStyle(vm.comfyRunning ? Color.green : (vm.comfyStarting ? Color.orange : .secondary))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(Capsule().fill(Color.primary.opacity(0.06)))
            )
        ) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    if vm.comfyRunning && !vm.comfyPublicURL.isEmpty {
                        Button {
                            vm.openComfyUIInBrowser()
                        } label: {
                            HStack(spacing: 7) {
                                Image(systemName: "safari.fill")
                                Text(L("在浏览器打开全屏节点画布", "Open Fullscreen Canvas in Browser"))
                                    .font(.headline)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)

                        Button {
                            copyToPasteboard(
                                vm.comfyPublicURL,
                                tip: L("已复制 ComfyUI 公网画布地址到剪贴板", "Copied ComfyUI public canvas URL to clipboard")
                            )
                        } label: {
                            Label(L("复制画布链接", "Copy Canvas URL"), systemImage: "doc.on.doc")
                                .padding(.vertical, 8)
                        }
                        .controlSize(.large)

                        Button {
                            Task { await vm.startRemoteComfyUI(forceRestart: true, autoOpenBrowser: false) }
                        } label: {
                            Label(L("重启生效", "Restart"), systemImage: "arrow.clockwise")
                                .padding(.vertical, 8)
                        }
                        .controlSize(.large)
                        .disabled(vm.comfyStarting)
                        .help(L(
                            "在 Manager 安装新自定义节点后可点击一键重启并自动持久化至云盘",
                            "Restart ComfyUI after installing custom nodes in Manager and auto-sync to Google Drive"
                        ))
                    } else {
                        Button {
                            Task { await vm.startRemoteComfyUI(forceRestart: false, autoOpenBrowser: true) }
                        } label: {
                            HStack(spacing: 8) {
                                if vm.comfyStarting {
                                    ProgressView().controlSize(.small)
                                } else {
                                    Image(systemName: "bolt.circle.fill")
                                }
                                Text(vm.comfyStarting
                                     ? L("正在部署云端 ComfyUI 并恢复 Google Drive 软链与节点...",
                                         "Deploying Cloud ComfyUI & restoring Google Drive symlinks and custom nodes...")
                                     : (vm.isConnected
                                        ? L("一键拉起云端 ComfyUI 并在浏览器打开全屏画布", "Deploy & Open Fullscreen Cloud ComfyUI Canvas")
                                        : L("请先在左侧唤醒或连接 G4 实例", "Wake or connect a G4 instance on the left first")))
                                    .font(.headline)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .disabled(!vm.isConnected || vm.comfyStarting)
                    }
                }

                if vm.comfyStarting || vm.comfyDownloadingWeight {
                    ProgressView(value: vm.comfyProgress, total: 1.0)
                }

                HStack(spacing: 8) {
                    Image(systemName: vm.comfyRunning ? "checkmark.shield.fill" : "info.circle")
                        .font(.system(size: 12))
                        .foregroundStyle(vm.comfyRunning ? Color.green : Color.accentColor)
                    Text(vm.displayComfyStatusMessage)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    Spacer()
                    if vm.comfyRunning && !vm.comfyPublicURL.isEmpty {
                        Text(vm.maskedURLDisplay(vm.comfyPublicURL))
                            .font(.system(size: 11.5, design: .monospaced))
                            .foregroundStyle(Color.accentColor)
                            .textSelection(.enabled)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(RoundedRectangle(cornerRadius: 8).fill(palette.inputFill))
            }
        }

        // 2. 新手零门槛预置图生视频 / 首尾帧运镜工作流包卡片
        WorkspaceCard(
            title: L(
                "预置图生视频 / 首尾帧运镜工作流包（一键配齐模型与中文画布）",
                "Preset Image-to-Video & First/Last Frame Workflow Packs (1-Click Setup)"
            ),
            subtitle: L(
                "专为零基础设计：一键 8 线程下齐配套权重、自动生成全中文三步注释工作流并将 ComfyUI 设为简体中文，零红框报错直接出片",
                "Turnkey setup: 8-thread download of all required weights plus annotated 3-step workflow templates — zero missing-node errors"
            ),
            icon: "wand.and.stars",
            trailing: AnyView(
                Text(L(
                    "已就绪 \(vm.comfyPresetPacks.filter(\.ready).count) / \(vm.comfyPresetPacks.count) 套",
                    "Ready \(vm.comfyPresetPacks.filter(\.ready).count) / \(vm.comfyPresetPacks.count)"
                ))
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(vm.comfyPresetPacks.contains(where: \.ready) ? Color.green : .secondary)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 3.5)
                    .background(Capsule().fill(Color.primary.opacity(0.06)))
            )
        ) {
            VStack(alignment: .leading, spacing: 12) {
                // 三步傻瓜引导条
                HStack(spacing: 8) {
                    comfyStepChip(
                        num: "1",
                        title: L("点击下方「一键配齐模型与工作流」", "Click '1-Click Install Pack' below"),
                        icon: "arrow.down.circle.fill",
                        palette: palette
                    )
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.secondary.opacity(0.6))
                    comfyStepChip(
                        num: "2",
                        title: L("在画布最左侧点「工作流」选中文模板", "Open 'Workflows' sidebar & pick template"),
                        icon: "sidebar.left",
                        palette: palette
                    )
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.secondary.opacity(0.6))
                    comfyStepChip(
                        num: "3",
                        title: L("上传 1~2 张图片点右上角「运行」出片", "Upload 1–2 images & click 'Run'"),
                        icon: "play.circle.fill",
                        palette: palette
                    )
                }

                VStack(spacing: 10) {
                    ForEach(vm.comfyPresetPacks) { pack in
                        comfyPresetPackRow(pack, palette: palette)
                    }
                }
            }
        }

        // 3. Google Drive 自动软链持久化矩阵卡片
        WorkspaceCard(
            title: L("Google Drive 自动软链与跨实例持久化", "Google Drive Auto-Symlink Persistence Across Instances"),
            subtitle: L(
                "新开 G4 实例自动挂载恢复自定义节点（含自动补齐 Python requirements.txt 依赖）、模型权重与工作流，免重复安装",
                "Fresh G4 runtimes automatically restore custom nodes (including Python requirements.txt), model weights, and saved workflows"
            ),
            icon: "externaldrive.badge.icloud",
            trailing: AnyView(
                Button {
                    Task { await vm.syncRemoteComfyUIToDrive() }
                } label: {
                    HStack(spacing: 4) {
                        if vm.comfySyncingDrive {
                            ProgressView().controlSize(.mini)
                        } else {
                            Image(systemName: "arrow.triangle.2.circlepath")
                        }
                        Text(vm.comfySyncingDrive ? L("正在同步...", "Syncing...") : L("立即同步至云盘", "Sync to Drive Now"))
                    }
                }
                .controlSize(.small)
                .disabled(!vm.isConnected || vm.comfySyncingDrive)
            )
        ) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    comfyPersistMetricTile(
                        title: L("models/* 模型权重库", "models/* Weight Library"),
                        value: L("\(vm.comfyModelsCount) 个权重文件", "\(vm.comfyModelsCount) weight files"),
                        detail: L("各子目录自动软链至云盘，新节点自建目录自动接管", "Subfolders symlinked to Drive; new node folders auto-adopted"),
                        icon: "cube.box.fill",
                        palette: palette
                    )
                    comfyPersistMetricTile(
                        title: L("custom_nodes 自定义节点", "custom_nodes Extensions"),
                        value: L("\(vm.comfyCustomNodesCount) 个节点插件", "\(vm.comfyCustomNodesCount) custom nodes"),
                        detail: L("NVMe 高速运行 + 云盘双向镜像 + 自动重装依赖", "Local NVMe speed + two-way Drive mirror + auto pip deps"),
                        icon: "puzzlepiece.extension.fill",
                        palette: palette
                    )
                    comfyPersistMetricTile(
                        title: L("workflows 工作流存档", "workflows Saved Graphs"),
                        value: L("\(vm.comfyWorkflowsCount) 份工作流", "\(vm.comfyWorkflowsCount) workflows"),
                        detail: L("画布 Ctrl+S 保存即自动持久化至 Google Drive", "Saving with Ctrl+S in canvas persists directly to Google Drive"),
                        icon: "flowchart.fill",
                        palette: palette
                    )
                }

                HStack(spacing: 6) {
                    Text(L("云端持久化根路径:", "Drive Persistence Root:"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(vm.comfyPersistRoot)
                        .font(.system(size: 11.5, design: .monospaced))
                        .foregroundStyle(.primary)
                        .textSelection(.enabled)
                    Spacer()
                    Text(vm.displayComfyPersistenceLabel)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(vm.comfyDrivePersisted ? Color.green : .secondary)
                }
            }
        }

        // 4. 8 线程极速下载权重直达 ComfyUI 云盘目录
        WorkspaceCard(
            title: L("8 线程极速下载权重至 ComfyUI 云盘目录", "8-Thread High-Speed Weight Downloader to ComfyUI Drive"),
            subtitle: L(
                "比浏览器单线程快 5~10 倍，下载完成后自动软链落盘至 Google Drive；在 ComfyUI 画布按 R 键即可刷新选用",
                "5–10× faster than single-thread browser downloads, persisted to Google Drive via symlinks; press R in ComfyUI to refresh node dropdowns"
            ),
            icon: "arrow.down.circle.dotted"
        ) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(L("目标模型目录 (ComfyUI/models/...)", "Target Subfolder (ComfyUI/models/...)"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Picker(L("目标目录", "Target Folder"), selection: $vm.comfyWeightSubfolder) {
                            ForEach(modelSubfolders, id: \.0) { item in
                                Text(item.1).tag(item.0)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 285)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text(L("HuggingFace 直链 / 下载 URL / 仓库 ID", "HuggingFace Direct URL / Download Link / Repo ID"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        TextField(
                            L("URL 或 org/repo", "URL or org/repo"),
                            text: $vm.comfyWeightSourceInput,
                            prompt: Text(L(
                                "如 https://huggingface.co/.../resolve/main/xxx.safetensors",
                                "e.g. https://huggingface.co/.../resolve/main/xxx.safetensors"
                            ))
                        )
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12, design: .monospaced))
                    }
                }

                HStack(spacing: 10) {
                    TextField(
                        L("可选文件名或仓库内子路径", "Optional filename or repo subpath"),
                        text: $vm.comfyWeightFilenameInput,
                        prompt: Text(L(
                            "直链可留空；填仓库 ID 时填仓库内文件路径（如 split_files/vae/xxx.safetensors）",
                            "Leave blank for direct URLs; when entering a Repo ID, specify subpath (e.g. split_files/vae/xxx.safetensors)"
                        ))
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12, design: .monospaced))

                    Button {
                        Task { await vm.startComfyWeightDownload() }
                    } label: {
                        HStack(spacing: 5) {
                            if vm.comfyDownloadingWeight {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "bolt.horizontal.fill")
                            }
                            Text(vm.comfyDownloadingWeight
                                 ? L("极速下载中...", "Downloading...")
                                 : L("8 线程下载至云盘", "8-Thread Download to Drive"))
                                .fontWeight(.semibold)
                        }
                        .padding(.horizontal, 10)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!vm.isConnected || vm.comfyDownloadingWeight)
                }
            }
        }

        // 5. ComfyUI 云端输出成片库 + 本地预览联动
        HStack(alignment: .top, spacing: 14) {
            mediaPreviewCard

            WorkspaceCard(
                title: L("ComfyUI 输出成片", "ComfyUI Cloud Outputs"),
                subtitle: L("点击一键拉取至 Mac 本地", "Click to pull renders to local Mac"),
                icon: "film.stack"
            ) {
                if vm.comfyRecentOutputs.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "video.badge.plus")
                            .font(.system(size: 22))
                            .foregroundStyle(.secondary.opacity(0.6))
                        Text(L(
                            "在 ComfyUI 画布渲染生成的视频与图片会自动显示在此处，并同步保存至 Google Drive",
                            "Videos and images rendered in the ComfyUI canvas appear here automatically and sync to Google Drive"
                        ))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 28)
                } else {
                    ScrollView {
                        VStack(spacing: 6) {
                            ForEach(vm.comfyRecentOutputs) { item in
                                HStack(spacing: 8) {
                                    Image(systemName: item.media_type == "video" ? "film" : "photo")
                                        .font(.system(size: 12, weight: .semibold))
                                        .foregroundStyle(Color.accentColor)
                                        .frame(width: 20)
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(item.filename)
                                            .font(.system(size: 11.5, weight: .medium))
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                        Text("\(fmt(item.size_mb ?? 0, 1)) MB · \(item.media_type == "video" ? L("视频", "Video") : L("图像", "Image"))")
                                            .font(.system(size: 10))
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer(minLength: 4)
                                    Button {
                                        Task { await vm.downloadComfyOutputToMac(item) }
                                    } label: {
                                        if vm.comfyPullingOutputRelPath == item.rel_path {
                                            ProgressView().controlSize(.mini)
                                        } else {
                                            Image(systemName: "arrow.down.to.line.compact")
                                                .font(.system(size: 12, weight: .semibold))
                                        }
                                    }
                                    .buttonStyle(.bordered)
                                    .controlSize(.small)
                                    .disabled(vm.comfyPullingOutputRelPath != nil)
                                    .help(L("拉取到 Mac 本地目录并在左侧播放器预览", "Pull to local Mac folder & preview in player"))
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 6)
                                .background(RoundedRectangle(cornerRadius: 7).fill(palette.inputFill))
                            }
                        }
                    }
                    .frame(maxHeight: 280)
                }
            }
            .frame(width: 290)
        }
    }

    func comfyPersistMetricTile(
        title: String,
        value: String,
        detail: String,
        icon: String,
        palette: StudioPalette
    ) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
                Text(title)
                    .font(.system(size: 11.5, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Text(value)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
            Text(detail)
                .font(.system(size: 10.5))
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .padding(11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 9).fill(palette.inputFill))
        .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(palette.cardBorder, lineWidth: 0.7))
    }

    func comfyStepChip(
        num: String,
        title: String,
        icon: String,
        palette: StudioPalette
    ) -> some View {
        HStack(spacing: 6) {
            Text(num)
                .font(.system(size: 10.5, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
                .frame(width: 18, height: 18)
                .background(Circle().fill(Color.accentColor))
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Color.accentColor)
            Text(title)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(.primary)
                .lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(palette.inputFill))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(palette.cardBorder, lineWidth: 0.7))
    }

    func comfyPresetPackRow(_ pack: ComfyPresetPackItem, palette: StudioPalette) -> some View {
        let isInstallingThis = vm.comfyInstallingPackId == pack.id
        let isAnyBusy = vm.comfyDownloadingWeight || !vm.comfyInstallingPackId.isEmpty || vm.comfyStarting
        let isPrimaryRecommended = pack.id == "wan22_i2v_flf2v"

        return VStack(alignment: .leading, spacing: 9) {
            HStack(alignment: .top, spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 9)
                        .fill(pack.ready ? Color.green.opacity(0.14) : Color.accentColor.opacity(0.12))
                        .frame(width: 36, height: 36)
                    Image(systemName: pack.ready ? "checkmark.seal.fill" : (pack.id.contains("minimax") || pack.id.contains("ltx") ? "waveform.and.magnifyingglass" : "film.stack.fill"))
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(pack.ready ? Color.green : Color.accentColor)
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 7) {
                        Text(pack.localizedTitle)
                            .font(.system(size: 13.5, weight: .bold))
                            .foregroundStyle(.primary)

                        Text(pack.localizedBadge)
                            .font(.system(size: 10.5, weight: .semibold))
                            .foregroundStyle(isPrimaryRecommended ? Color.white : Color.accentColor)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background(
                                Capsule().fill(isPrimaryRecommended ? Color.accentColor : Color.accentColor.opacity(0.13))
                            )

                        if pack.ready {
                            Text(L("已配齐就绪", "Ready"))
                                .font(.system(size: 10.5, weight: .semibold))
                                .foregroundStyle(Color.green)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 2)
                                .background(Capsule().fill(Color.green.opacity(0.13)))
                        }
                    }

                    Text(pack.localizedScenario)
                        .font(.system(size: 11.5))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    HStack(spacing: 6) {
                        Label(
                            L("权重 \(pack.installed_models)/\(pack.total_models) · 约 \(fmt(pack.approx_size_gb, 1)) GB",
                              "Weights \(pack.installed_models)/\(pack.total_models) · ~\(fmt(pack.approx_size_gb, 1)) GB"),
                            systemImage: "internaldrive"
                        )
                            .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)

                        Text("·")
                            .foregroundStyle(.secondary)

                        ForEach(pack.workflows, id: \.self) { wfName in
                            Text(ComfyPresetPackItem.localizedWorkflowLabel(wfName))
                                .font(.system(size: 10.5, weight: .medium))
                                .foregroundStyle(.primary.opacity(0.85))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(RoundedRectangle(cornerRadius: 5).fill(Color.primary.opacity(0.06)))
                                .lineLimit(1)
                        }
                    }
                    .padding(.top, 1)
                }

                Spacer(minLength: 8)

                VStack(alignment: .trailing, spacing: 6) {
                    if pack.ready {
                        HStack(spacing: 6) {
                            Button {
                                Task { await vm.installComfyPresetPack(pack, openWhenReady: false) }
                            } label: {
                                Text(L("重写中文模板", "Rebuild Templates"))
                                    .font(.system(size: 11.5))
                            }
                            .controlSize(.small)
                            .disabled(!vm.isConnected || isAnyBusy)
                            .help(L(
                                "重新生成该工作流包的标准中文傻瓜注释画布文件",
                                "Regenerate annotated workflow JSON templates for this pack"
                            ))

                            Button {
                                if vm.comfyRunning && !vm.comfyPublicURL.isEmpty {
                                    vm.openComfyUIInBrowser()
                                } else {
                                    Task { await vm.startRemoteComfyUI(forceRestart: false, autoOpenBrowser: true) }
                                }
                            } label: {
                                HStack(spacing: 5) {
                                    Image(systemName: "safari.fill")
                                    Text(L("打开画布直接使用", "Open in Canvas"))
                                        .fontWeight(.semibold)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.green)
                            .disabled(!vm.isConnected || isAnyBusy)
                        }
                    } else {
                        Button {
                            Task { await vm.installComfyPresetPack(pack, openWhenReady: true) }
                        } label: {
                            HStack(spacing: 6) {
                                if isInstallingThis {
                                    ProgressView().controlSize(.small)
                                } else {
                                    Image(systemName: "bolt.badge.checkmark.fill")
                                }
                                Text(isInstallingThis
                                     ? L("正在极速配齐 (\(Int((vm.comfyProgress * 100).rounded()))%)...",
                                         "Installing (\(Int((vm.comfyProgress * 100).rounded()))%)...")
                                     : L("一键配齐模型与工作流", "1-Click Install Pack"))
                                    .fontWeight(.semibold)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!vm.isConnected || isAnyBusy)
                    }
                }
            }

            if isInstallingThis {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: vm.comfyProgress, total: 1.0)
                    Text(vm.displayComfyStatusMessage)
                        .font(.system(size: 11))
                        .foregroundStyle(Color.accentColor)
                        .lineLimit(1)
                }
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isPrimaryRecommended && !pack.ready
                      ? Color.accentColor.opacity(colorScheme == .dark ? 0.10 : 0.045)
                      : palette.inputFill)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(
                    pack.ready ? Color.green.opacity(0.45) : (isPrimaryRecommended ? Color.accentColor.opacity(0.45) : palette.cardBorder),
                    lineWidth: (isPrimaryRecommended || pack.ready) ? 1.1 : 0.7
                )
        )
    }

    // MARK: - OpenAI & Anthropic 双协议公网 API 服务网关面板

    func copyToPasteboard(_ text: String, tip: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        vm.statusMessage = tip
    }

    func apiGatewaySection(palette: StudioPalette) -> some View {
        let baseHost = vm.normalizedBaseURL.isEmpty ? "https://xxxx.gradio.live" : vm.normalizedBaseURL
        let openaiBase = "\(baseHost)/v1"
        let anthropicBase = baseHost
        let displayKey = vm.apiKeyEnabled ? vm.apiAccessKey : "sk-no-key-required"
        let activeLLMRepo: String = {
            if vm.selectedModelId == "qwen38_27b_gguf" { return "unsloth/Qwen3.8-27B-GGUF" }
            if vm.selectedModelId == "qwen38_27b" { return "Qwen/Qwen3.8-27B" }
            if vm.loadedModelIds.contains("qwen38_27b_gguf") && !vm.loadedModelIds.contains("qwen38_27b") {
                return "unsloth/Qwen3.8-27B-GGUF"
            }
            return "Qwen/Qwen3.8-27B"
        }()
        let activeLLMDisplayName = activeLLMRepo.contains("GGUF") ? "Qwen3.8-27B-GGUF" : "Qwen3.8-27B"

        let samplePrompt1 = L("用三句话解释量子纠缠，并展示思考过程", "Explain quantum entanglement in 3 sentences with reasoning")
        let samplePrompt2 = L("帮我设计一个高并发网关架构", "Design a high-concurrency API gateway architecture")
        let samplePrompt3 = L("分析 Rust 所有权机制的核心优势", "Analyze the core advantages of Rust's ownership model")

        let openaiCurl = """
        curl \(openaiBase)/chat/completions \\
          -H "Content-Type: application/json" \\
          -H "Authorization: Bearer \(displayKey)" \\
          -d '{
            "model": "\(activeLLMRepo)",
            "stream": true,
            "messages": [{"role": "user", "content": "\(samplePrompt1)"}]
          }'
        """

        let openaiPython = """
        from openai import OpenAI

        client = OpenAI(base_url="\(openaiBase)", api_key="\(displayKey)")
        stream = client.chat.completions.create(
            model="\(activeLLMRepo)",
            messages=[{"role": "user", "content": "\(samplePrompt2)"}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                print(delta.reasoning_content, end="", flush=True)
            if delta.content:
                print(delta.content, end="", flush=True)
        """

        let cursorConfig = """
        # \(L("Cursor -> Settings -> Models -> OpenAI API Key 配置", "Cursor -> Settings -> Models -> OpenAI API Key"))
        Override OpenAI Base URL: \(openaiBase)
        OpenAI API Key:           \(displayKey)
        Custom Model Name:        \(activeLLMRepo)
        """

        let anthropicCurl = """
        curl \(anthropicBase)/v1/messages \\
          -H "Content-Type: application/json" \\
          -H "x-api-key: \(displayKey)" \\
          -H "anthropic-version: 2023-06-01" \\
          -d '{
            "model": "\(activeLLMRepo)",
            "max_tokens": 2048,
            "stream": true,
            "messages": [{"role": "user", "content": "\(samplePrompt3)"}]
          }'
        """

        let claudeCodeEnv = """
        export ANTHROPIC_BASE_URL="\(anthropicBase)"
        export ANTHROPIC_API_KEY="\(displayKey)"
        export ANTHROPIC_MODEL="\(activeLLMRepo)"
        """

        return VStack(alignment: .leading, spacing: 16) {
            WorkspaceCard(
                title: L("网关安全鉴权与运行状态", "Gateway Authentication & Status"),
                subtitle: L(
                    "公网隧道开启期间，建议保持密钥鉴权开启以保护你的 Colab 算力单元 (CU)",
                    "Keep API Key authentication enabled while the public tunnel is live to protect your Colab Compute Units (CU)"
                ),
                icon: "lock.shield",
                trailing: AnyView(
                    Text(L("累计 API 请求: \(vm.apiRequestCount) 次", "Total API Requests: \(vm.apiRequestCount)"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                )
            ) {
                HStack(spacing: 12) {
                    Text(L("API Key 鉴权已启用", "API key authentication is required"))
                        .font(.callout.weight(.medium))

                    Spacer()

                    if !vm.isQwenLoadedInVRAM {
                        Text(L(
                            "提示：请先在左侧点击「加载到显存」装载 \(activeLLMDisplayName)（可与其他模型同时共驻显存）",
                            "Tip: Click 'Load to VRAM' on the left to load \(activeLLMDisplayName) first (co-resides with other models)"
                        ))
                            .font(.caption)
                            .foregroundStyle(.orange)
                    } else if vm.loadedModelIds.count > 1 {
                        Text(L(
                            "多模型共驻已激活：外部 API 可随时调用 \(activeLLMDisplayName)，与视频/生图模型零切换共存",
                            "Multi-model VRAM active: external clients can call \(activeLLMDisplayName) alongside video/image models with zero reload"
                        ))
                            .font(.caption)
                            .foregroundStyle(.green)
                    }
                }

                if vm.apiKeyEnabled {
                    HStack(spacing: 8) {
                        Text(L("访问密钥", "API Key"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        TextField("API Key", text: $vm.apiAccessKey)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12, design: .monospaced))
                            .onSubmit {
                                Task { await vm.syncApiKeyToColab() }
                            }
                        Button(L("同步到云端", "Sync to Cloud")) {
                            Task {
                                await vm.syncApiKeyToColab()
                                vm.statusMessage = L("API 密钥已同步更新至 Colab 云端网关。", "API Key synced to Colab cloud gateway.")
                            }
                        }
                        .controlSize(.small)
                        .disabled(!vm.isConnected)

                        Button(L("重新生成", "Regenerate")) {
                            vm.regenerateApiKey()
                            vm.statusMessage = L("已生成新的随机 API Key。", "Generated a new random API Key.")
                        }
                        .controlSize(.small)

                        Button(L("复制密钥", "Copy Key")) {
                            copyToPasteboard(vm.apiAccessKey, tip: L("已复制 API Key 到剪贴板。", "Copied API Key to clipboard."))
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                }
            }

            HStack(alignment: .top, spacing: 16) {
                // OpenAI 协议卡片
                WorkspaceCard(
                    title: L("OpenAI 兼容协议", "OpenAI-Compatible Protocol (`/v1`)"),
                    subtitle: L(
                        "兼容 OpenAI SDK、Cursor、Cherry Studio、Chatbox 等客户端",
                        "Compatible with OpenAI SDK, Cursor, Cherry Studio, Chatbox & more"
                    ),
                    icon: "bolt.horizontal.circle"
                ) {
                    endpointRow(label: "Base URL", value: openaiBase, tip: L("已复制 OpenAI Base URL", "Copied OpenAI Base URL"))
                    endpointRow(label: L("Chat 接口", "Chat URL"), value: "\(openaiBase)/chat/completions", tip: L("已复制 Chat Completions 地址", "Copied Chat Completions URL"))
                    endpointRow(label: L("模型名称", "Model ID"), value: activeLLMRepo, tip: L("已复制模型名称", "Copied Model ID"))

                    Text(L(
                        "支持 stream 流式输出、reasoning_content 深度思考链、image_url 视觉输入与 Function Calling 工具调用；加载图像模型时亦支持 /v1/images/generations。",
                        "Supports SSE streaming, `reasoning_content` thinking chains, `image_url` vision inputs, and Function Calling; also serves `/v1/images/generations` when an image model is loaded."
                    ))
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Picker("", selection: $openaiSnippetTab) {
                        Text(L("Python 示例", "Python SDK")).tag(0)
                        Text("Cursor").tag(1)
                        Text("curl").tag(2)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .controlSize(.small)

                    codeSnippetBlock(
                        title: openaiSnippetTab == 0
                            ? L("Python OpenAI SDK 流式示例", "Python OpenAI SDK Streaming Example")
                            : (openaiSnippetTab == 1 ? L("Cursor 编辑器接入配置", "Cursor IDE Custom Model Setup") : "OpenAI Chat Completions curl"),
                        code: openaiSnippetTab == 0 ? openaiPython : (openaiSnippetTab == 1 ? cursorConfig : openaiCurl),
                        palette: palette
                    )

                    HStack(spacing: 8) {
                        Button(L("复制 Python 代码", "Copy Python")) {
                            copyToPasteboard(openaiPython, tip: L("已复制 OpenAI Python 调用代码。", "Copied OpenAI Python snippet."))
                        }
                        .controlSize(.small)

                        Button(L("复制 Cursor 配置", "Copy Cursor")) {
                            copyToPasteboard(cursorConfig, tip: L("已复制 Cursor 接入配置。", "Copied Cursor configuration."))
                        }
                        .controlSize(.small)

                        Button(L("复制 curl 命令", "Copy curl")) {
                            copyToPasteboard(openaiCurl, tip: L("已复制 OpenAI curl 测试命令。", "Copied OpenAI curl command."))
                        }
                        .controlSize(.small)
                    }
                }

                // Anthropic 协议卡片
                WorkspaceCard(
                    title: L("Anthropic Messages 协议", "Anthropic Messages Protocol (`/v1/messages`)"),
                    subtitle: L(
                        "原生实现 /v1/messages，兼容 Anthropic SDK 与 Claude Code",
                        "Native `/v1/messages` endpoint for Anthropic SDK & Claude Code"
                    ),
                    icon: "chevron.left.forwardslash.chevron.right"
                ) {
                    endpointRow(label: "Base URL", value: anthropicBase, tip: L("已复制 Anthropic Base URL", "Copied Anthropic Base URL"))
                    endpointRow(label: "Messages", value: "\(anthropicBase)/v1/messages", tip: L("已复制 Anthropic Messages 地址", "Copied Anthropic Messages URL"))
                    endpointRow(label: L("鉴权请求头", "Auth Header"), value: L("x-api-key 或 Authorization: Bearer", "x-api-key or Authorization: Bearer"), tip: L("已复制请求头说明", "Copied auth header format"))

                    Text(L(
                        "完整支持 Anthropic SSE 事件流（thinking_delta 思考块、text_delta 文本块、tool_use 工具块）及 /v1/messages/count_tokens 接口，任意模型名均可自动路由至 \(activeLLMDisplayName)。",
                        "Full Anthropic SSE streaming (`thinking_delta`, `text_delta`, `tool_use`) plus `/v1/messages/count_tokens`; any model alias routes automatically to \(activeLLMDisplayName)."
                    ))
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Picker("", selection: $anthropicSnippetTab) {
                        Text("Claude Code").tag(0)
                        Text("curl").tag(1)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .controlSize(.small)

                    codeSnippetBlock(
                        title: anthropicSnippetTab == 0
                            ? L("Claude Code / 终端环境变量接入", "Claude Code / Shell Environment Variables")
                            : "Anthropic /v1/messages curl",
                        code: anthropicSnippetTab == 0 ? claudeCodeEnv : anthropicCurl,
                        palette: palette
                    )

                    HStack(spacing: 8) {
                        Button(L("复制环境变量配置", "Copy Claude Code Env")) {
                            copyToPasteboard(claudeCodeEnv, tip: L("已复制 ANTHROPIC_BASE_URL 环境变量配置。", "Copied ANTHROPIC_BASE_URL environment variables."))
                        }
                        .controlSize(.small)

                        Button(L("复制 Anthropic curl", "Copy Anthropic curl")) {
                            copyToPasteboard(anthropicCurl, tip: L("已复制 Anthropic /v1/messages curl 命令。", "Copied Anthropic /v1/messages curl command."))
                        }
                        .controlSize(.small)
                    }
                }
            }
        }
    }

    func endpointRow(label: String, value: String, tip: String) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 72, alignment: .leading)
            Text(value)
                .font(.system(size: 11.5, design: .monospaced))
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button(L("复制", "Copy")) {
                copyToPasteboard(value, tip: tip)
            }
            .controlSize(.mini)
        }
        .padding(.vertical, 2)
    }

    func codeSnippetBlock(title: String, code: String, palette: StudioPalette) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            ScrollView(.horizontal) {
                Text(code)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Color(white: 0.88))
                    .textSelection(.enabled)
                    .padding(10)
            }
            .environment(\.colorScheme, .dark)
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(height: 122)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color(red: 0.08, green: 0.09, blue: 0.11)))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    // MARK: - 优化后的醒目模式切换器 (Mode Switcher Bar)

    func modeSwitcherHeader(palette: StudioPalette) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 10) {
                Image(systemName: vm.activeModelSpec.icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
                    .frame(width: 32, height: 32)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.accentColor.opacity(0.12)))

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text(vm.activeModelSpec.name)
                            .font(.title3.weight(.bold))
                        Text(vm.activeModelSpec.repoId)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.primary.opacity(0.06)))
                    }
                    Text(vm.activeModelSpec.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                HStack(spacing: 8) {
                    if vm.loadedModelIds.count > 1 {
                        HStack(spacing: 4) {
                            Image(systemName: "square.stack.3d.up.fill")
                                .font(.system(size: 10, weight: .semibold))
                            Text(L("\(vm.loadedModelIds.count) 模型显存共驻", "\(vm.loadedModelIds.count) Models in VRAM"))
                                .font(.caption2.weight(.semibold))
                        }
                        .foregroundStyle(Color.accentColor)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(Capsule().fill(Color.accentColor.opacity(0.12)))
                    }

                    HStack(spacing: 6) {
                        StatusDot(color: vm.isSelectedModelLoadedInVRAM ? .green : .secondary.opacity(0.5))
                        Text(vm.isSelectedModelLoadedInVRAM
                             ? L("显存已就绪", "Ready in VRAM")
                             : L("待加载至显存", "Not Loaded in VRAM"))
                            .font(.caption.weight(.medium))
                            .foregroundStyle(vm.isSelectedModelLoadedInVRAM ? Color.green : Color.secondary)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Capsule().fill(palette.inputFill))
                    .overlay(Capsule().strokeBorder(palette.cardBorder, lineWidth: 0.8))
                }
            }

            // 双卡片式高辨识度模式切换栏
            HStack(spacing: 12) {
                ForEach(vm.activeModelSpec.modes) { mode in
                    let isSelected = vm.taskModeId == mode.id
                    Button {
                        withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                            vm.taskModeId = mode.id
                        }
                    } label: {
                        HStack(spacing: 12) {
                            ZStack {
                                RoundedRectangle(cornerRadius: 9)
                                    .fill(isSelected ? Color.accentColor : Color.primary.opacity(0.06))
                                    .frame(width: 38, height: 38)
                                Image(systemName: mode.icon)
                                    .font(.system(size: 16, weight: .semibold))
                                    .foregroundStyle(isSelected ? Color.white : Color.secondary)
                            }

                            VStack(alignment: .leading, spacing: 3) {
                                HStack(spacing: 6) {
                                    Text(mode.title)
                                        .font(.system(size: 14, weight: .bold))
                                        .foregroundStyle(isSelected ? Color.primary : Color.secondary)
                                    if isSelected {
                                        Text(L("当前模式", "Active"))
                                            .font(.system(size: 10, weight: .semibold))
                                            .foregroundStyle(Color.accentColor)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 1.5)
                                            .background(Capsule().fill(Color.accentColor.opacity(0.14)))
                                    }
                                }
                                Text(mode.subtitle)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }

                            Spacer(minLength: 0)

                            Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                                .font(.system(size: 15, weight: .medium))
                                .foregroundStyle(isSelected ? Color.accentColor : Color.secondary.opacity(0.45))
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(isSelected ? Color.accentColor.opacity(colorScheme == .dark ? 0.16 : 0.07) : palette.inputFill)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .strokeBorder(
                                    isSelected ? Color.accentColor : palette.cardBorder,
                                    lineWidth: isSelected ? 1.6 : 0.8
                                )
                        )
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(palette.workspaceCardFill)
                .shadow(color: .black.opacity(colorScheme == .dark ? 0.25 : 0.04), radius: 4, x: 0, y: 1.5)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(palette.cardBorder, lineWidth: 0.8)
        )
    }

    @ViewBuilder
    var referenceImageSection: some View {
        switch vm.taskModeId {
        case "fl2va":
            WorkspaceCard(
                title: L("首尾关键帧参考图", "Start & End Keyframe Images"),
                subtitle: L("至少拖入首帧或尾帧之一，精准锁定起止画面构图", "Drop at least a start frame or end frame to anchor camera motion & composition"),
                icon: "photo.on.rectangle.angled"
            ) {
                HStack(spacing: 12) {
                    FrameDropWell(
                        title: L("起始首帧 (Index 0)", "Start Frame (Index 0)"),
                        subtitle: L("拖入图片或点击选择", "Drop an image or click to browse"),
                        image: $vm.firstFrameImage
                    )
                    FrameDropWell(
                        title: L("结束尾帧 (可选 · Index -1)", "End Frame (Optional · Index -1)"),
                        subtitle: L("拖入图片或点击选择", "Drop an image or click to browse"),
                        image: $vm.lastFrameImage
                    )
                }
            }
        case "i2v":
            WorkspaceCard(
                title: L("起始首帧参考图", "Start Keyframe Image"),
                subtitle: L("模型将以该图片作为视频第 1 帧向后延展动态画面", "The model animates forward starting from this image as frame 0"),
                icon: "photo.badge.arrow.down"
            ) {
                FrameDropWell(
                    title: L("首帧原图", "Start Frame Image"),
                    subtitle: L("拖入本地图片或点击浏览", "Drop a local image or click to browse"),
                    image: $vm.firstFrameImage
                )
            }
        case "i2i_edit":
            WorkspaceCard(
                title: (vm.selectedModelId == "qwen_image_21" || vm.isMiniMaxH3ImageModel)
                    ? L("指令修图参考原图", "Instruction Image-Editing References")
                    : L("图生图重绘参考底图", "Image-to-Image Reference"),
                subtitle: vm.isMiniMaxH3ImageModel
                    ? L("支持上传 1~2 张参考图（可在提示词中用 <Picture 1> / <Picture 2> 指代）进行精准修图或融合", "Upload 1–2 reference images (refer via <Picture 1> / <Picture 2> in prompt) for editing or fusion")
                    : (vm.selectedModelId == "qwen_image_21"
                        ? L("支持上传 1~2 张参考图进行局部编辑、换背景或多主体融合", "Upload 1–2 reference images for local edits, background swaps, or multi-subject fusion")
                        : L("上传参考原图并在左侧调节重绘幅度", "Upload a reference image and tune Denoising Strength in the left sidebar")),
                icon: "wand.and.rays"
            ) {
                HStack(spacing: 12) {
                    FrameDropWell(
                        title: L("主参考图 1 (必选)", "Primary Reference 1 (Required)"),
                        subtitle: L("拖入待编辑原图", "Drop source image to edit"),
                        image: $vm.firstFrameImage
                    )
                    if vm.selectedModelId == "qwen_image_21" || vm.isMiniMaxH3ImageModel {
                        FrameDropWell(
                            title: L("融合参考图 2 (可选)", "Fusion Reference 2 (Optional)"),
                            subtitle: L("用于多图主体融合编辑", "For multi-image subject fusion"),
                            image: $vm.lastFrameImage
                        )
                    }
                }
            }
        case "vlm":
            WorkspaceCard(
                title: L("待分析视觉图像", "Vision Reference Images"),
                subtitle: L(
                    "支持拖入 1~2 张截图、摄影或分镜画稿交由 \(vm.activeModelSpec.name) 解析",
                    "Drop 1–2 screenshots, photos, or storyboard sketches for \(vm.activeModelSpec.name) to analyze"
                ),
                icon: "eye"
            ) {
                HStack(spacing: 12) {
                    FrameDropWell(
                        title: L("输入图像 1", "Input Image 1"),
                        subtitle: L("拖入图片进行解析或反推提示词", "Drop an image to analyze or reverse-engineer prompts"),
                        image: $vm.firstFrameImage
                    )
                    FrameDropWell(
                        title: L("对比图像 2 (可选)", "Comparison Image 2 (Optional)"),
                        subtitle: L("可同时传入第二张图像对比分析", "Optionally attach a second image for side-by-side comparison"),
                        image: $vm.lastFrameImage
                    )
                }
            }
        default:
            EmptyView()
        }
    }

    func promptComposerSection(palette: StudioPalette) -> some View {
        let mod = vm.activeModelSpec.modality
        let cardTitle: String = {
            switch mod {
            case .llmVLM:
                return L("问题与指令输入", "Message & Instruction Input")
            case .image:
                return vm.taskModeId == "i2i_edit"
                    ? L("修图指令与画面描述", "Editing Instruction & Scene Prompt")
                    : L("图像画面提示词", "Image Prompt")
            case .videoAudio, .video, .comfyUI:
                return L("镜头与多模态提示词", "Camera & Multimodal Prompts")
            }
        }()

        let primaryPlaceholder: String = {
            switch mod {
            case .llmVLM:
                return L("输入你想提问、推理编码或反推提示词的内容...", "Enter a question, coding task, or vision prompt...")
            case .image:
                return vm.taskModeId == "i2i_edit"
                    ? L("描述你想修改的细节（例如：把背景换成夕阳海滩，保留人物光影）...", "Describe your edit (e.g. replace background with a sunset beach while preserving lighting)...")
                    : L("描述主体、构图、镜头景别、光影色调与文字排版...", "Describe subject, composition, lighting, lens style, and typography...")
            case .videoAudio, .video, .comfyUI:
                return L("描述画面主体运动、镜头推拉摇移、光影氛围与电影质感...", "Describe subject motion, camera movement, lighting, and cinematic style...")
            }
        }()

        return WorkspaceCard(title: cardTitle, icon: "text.alignleft") {
            promptEditor(
                title: mod == .llmVLM
                    ? L("对话内容 / 视觉提问", "Chat Message / Vision Question")
                    : (vm.taskModeId == "i2i_edit"
                       ? L("编辑指令或目标画面描述", "Edit Instruction or Target Scene")
                       : L("主体、构图、光影与风格描述", "Subject, Composition, Lighting & Style")),
                placeholder: primaryPlaceholder,
                text: $vm.visualPrompt,
                height: mod == .llmVLM ? 84 : 92,
                palette: palette
            )

            if mod == .videoAudio {
                HStack(alignment: .top, spacing: 12) {
                    promptEditor(
                        title: L("环境音效与台词对白（可选）", "Soundscape, Foley & Dialogue (Optional)"),
                        placeholder: L("如：海浪拍岸声、脚步声、角色轻声说台词...", "e.g. Ocean waves crashing, footsteps on gravel, spoken dialogue..."),
                        text: $vm.soundscapePrompt,
                        height: 58,
                        palette: palette
                    )
                    promptEditor(
                        title: L("背景音乐风格（可选）", "Background Music Style (Optional)"),
                        placeholder: L("如：温暖大提琴与原声吉他电影配乐...", "e.g. Warm cinematic cello and acoustic guitar score..."),
                        text: $vm.musicPrompt,
                        height: 58,
                        palette: palette
                    )
                }

                DisclosureGroup(
                    L("反向提示词 Negative Prompt（可选，排除低质或多余元素）", "Negative Prompt (Optional — exclude artifacts or unwanted elements)"),
                    isExpanded: $showNegativePrompt
                ) {
                    promptEditor(
                        title: L("反向排除项", "Negative Terms"),
                        placeholder: L("如：模糊、畸变、水印、多余肢体、画面闪烁...", "e.g. blurry, distorted, watermark, extra limbs, flickering..."),
                        text: $vm.negativePrompt,
                        height: 44,
                        palette: palette
                    )
                    .padding(.top, 4)
                }
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
            } else if (mod == .image && !vm.isMiniMaxH3ImageModel) || mod == .video {
                promptEditor(
                    title: L("反向提示词 Negative Prompt（可选，排除低质或多余元素）", "Negative Prompt (Optional — exclude low quality or unwanted elements)"),
                    placeholder: L("如：模糊、低分辨率、畸变、水印、杂乱文字...", "e.g. blurry, low resolution, deformed, watermark, garbled text..."),
                    text: $vm.negativePrompt,
                    height: 46,
                    palette: palette
                )
            }
        }
    }

    func promptEditor(
        title: String,
        placeholder: String = "",
        text: Binding<String>,
        height: CGFloat,
        palette: StudioPalette
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
            ZStack(alignment: .topLeading) {
                if text.wrappedValue.isEmpty && !placeholder.isEmpty {
                    Text(placeholder)
                        .font(.body)
                        .foregroundStyle(.secondary.opacity(0.55))
                        .padding(.horizontal, 13)
                        .padding(.vertical, 10)
                        .allowsHitTesting(false)
                }
                TextEditor(text: text)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
            }
            .frame(height: height)
            .background(RoundedRectangle(cornerRadius: 8).fill(palette.inputFill))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(palette.cardBorder, lineWidth: 0.8))
        }
        .frame(maxWidth: .infinity)
    }

    var generateButtonTitle: String {
        let mod = vm.activeModelSpec.modality
        if vm.isGenerating {
            let liveTime = vm.currentJobElapsed > 0.05
                ? L(" · 已耗时 \(formatDuration(vm.currentJobElapsed, compact: true))",
                    " · \(formatDuration(vm.currentJobElapsed, compact: true))")
                : ""
            if mod == .llmVLM {
                return L(
                    "\(vm.activeModelSpec.name) 正在推理输出\(liveTime)...",
                    "\(vm.activeModelSpec.name) generating\(liveTime)..."
                )
            }
            return vm.batchCount > 1
                ? L("正在生成（共 \(vm.batchCount) 项\(liveTime)）...",
                    "Generating (\(vm.batchCount) items\(liveTime))...")
                : L("正在云端 GPU 生成\(liveTime)...",
                    "Generating on Cloud GPU\(liveTime)...")
        }
        if !vm.isSelectedModelLoadedInVRAM {
            return L(
                "开始运行（请先在左侧将 \(vm.activeModelSpec.name) 加载到显存）",
                "Run (Load \(vm.activeModelSpec.name) to VRAM on the left first)"
            )
        }
        switch mod {
        case .llmVLM:
            return L("发送并开始深度推理 (Command + Return)", "Send & Start Reasoning (Command + Return)")
        case .image:
            return vm.batchCount > 1
                ? L("生成 \(vm.batchCount) 张高清图像", "Generate \(vm.batchCount) HD Images")
                : L("开始生成图像", "Generate Image")
        case .videoAudio:
            return vm.batchCount > 1
                ? L("连续生成 \(vm.batchCount) 条全模态音视频", "Generate \(vm.batchCount) Videos with Audio")
                : L("开始生成音视频", "Generate Video + Audio")
        case .video, .comfyUI:
            return vm.batchCount > 1
                ? L("连续生成 \(vm.batchCount) 条高清视频", "Generate \(vm.batchCount) HD Videos")
                : L("开始生成视频", "Generate Video")
        }
    }

    var generateActionSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Button {
                    Task { await vm.startGeneration() }
                } label: {
                    HStack(spacing: 8) {
                        if vm.isGenerating {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: vm.activeModelSpec.modality == .llmVLM ? "paperplane.fill" : "play.fill")
                        }
                        Text(generateButtonTitle)
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 5)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(vm.isGenerating)
                .keyboardShortcut(.return, modifiers: [.command])

                if vm.isGenerating {
                    Button(role: .destructive) {
                        Task { await vm.cancelCurrentGeneration() }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "stop.fill")
                                .font(.caption)
                            Text(L("停止推理", "Stop"))
                                .font(.headline)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 5)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .help(L("立即中断当前云端推理任务并解锁操作", "Abort the running cloud GPU job immediately"))
                }
            }

            if vm.isGenerating {
                HStack(spacing: 8) {
                    ProgressView(value: vm.progressValue, total: 1.0)
                    Text("\(Int((vm.progressValue * 100).rounded()))%")
                        .font(.system(size: 11.5, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Color.accentColor)
                        .frame(width: 38, alignment: .trailing)
                }
            }

            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(vm.displayStatusMessage)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    // MARK: - Qwen3.8-27B / GGUF 专属流式思考与多轮对话面板

    func llmConversationCard(palette: StudioPalette) -> some View {
        WorkspaceCard(
            title: L(
                "\(vm.activeModelSpec.name) 深度推理与视觉对话输出",
                "\(vm.activeModelSpec.name) Deep Reasoning & Vision Chat"
            ),
            subtitle: L(
                "实时流式展示模型内部思考链与最终回答（当前上下文窗口 \(vm.contextLengthK)K），支持一键复制为视频/生图提示词",
                "Streams internal chain-of-thought and final response in real time (\(vm.contextLengthK)K context window)"
            ),
            icon: "bubble.left.and.text.bubble.right",
            trailing: AnyView(
                HStack(spacing: 8) {
                    Button {
                        withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                            vm.taskModeId = "api"
                        }
                    } label: {
                        Label(L("API 接入配置", "API Gateway"), systemImage: "network")
                    }
                    .controlSize(.small)

                    Button(L("清空对话", "Clear Chat")) {
                        vm.chatHistory.removeAll()
                    }
                    .controlSize(.small)
                    .disabled(vm.chatHistory.isEmpty || vm.isGenerating)
                }
            )
        ) {
            if vm.chatHistory.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "brain")
                        .font(.system(size: 30))
                        .foregroundStyle(.secondary.opacity(0.6))
                    Text(L(
                        "加载 \(vm.activeModelSpec.name) 后，在上方输入问题或拖入图片即可开启流式深度推理",
                        "Load \(vm.activeModelSpec.name), then enter a prompt or drop images above to stream reasoning & answers"
                    ))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 240)
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(vm.chatHistory) { turn in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 6) {
                                Image(systemName: turn.role == "user" ? "person.fill" : "sparkles")
                                    .font(.caption)
                                    .foregroundStyle(turn.role == "user" ? Color.secondary : Color.accentColor)
                                Text(turn.role == "user" ? L("提问", "You") : vm.activeModelSpec.name)
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(turn.role == "user" ? Color.secondary : Color.accentColor)
                                if turn.hasImage {
                                    Text(L("附带视觉图像", "Image Attached"))
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 1)
                                        .background(Capsule().fill(Color.primary.opacity(0.06)))
                                }
                                Spacer()
                                if turn.role == "assistant" && !turn.content.isEmpty {
                                    Button(L("复制回复", "Copy Reply")) {
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(turn.content, forType: .string)
                                    }
                                    .controlSize(.mini)
                                }
                            }

                            if !turn.thinking.isEmpty {
                                DisclosureGroup(L(
                                    "深度思考过程 (\(turn.thinking.count) 字)",
                                    "Chain of Thought (\(turn.thinking.count) chars)"
                                )) {
                                    Text(turn.thinking)
                                        .font(.system(size: 12))
                                        .foregroundStyle(.secondary)
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .padding(.top, 4)
                                }
                                .font(.caption.weight(.medium))
                                .padding(10)
                                .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.04)))
                            }

                            if !turn.content.isEmpty {
                                Text(turn.content)
                                    .font(.body)
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            } else if turn.role == "assistant" && vm.isGenerating {
                                HStack(spacing: 6) {
                                    ProgressView().controlSize(.small)
                                    Text(L("正在思考并组织回答...", "Thinking and composing response..."))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 9)
                                .fill(turn.role == "user" ? palette.inputFill : palette.workspaceCardFill)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 9)
                                .strokeBorder(palette.cardBorder, lineWidth: 0.7)
                        )
                    }
                }
            }
        }
    }

    // MARK: - 多媒体预览与历史面板（兼容视频播放与高清图像展示）

    var mediaPreviewCard: some View {
        let curElapsed = vm.elapsedForMedia(vm.currentMediaURL)
        return WorkspaceCard(
            title: L("生成结果预览", "Output Preview"),
            icon: "play.rectangle",
            trailing: curElapsed != nil ? AnyView(
                HStack(spacing: 4) {
                    Image(systemName: "stopwatch.fill")
                        .font(.system(size: 10.5, weight: .semibold))
                    Text(L("生成耗时 \(formatDuration(curElapsed!))", "Elapsed \(formatDuration(curElapsed!))"))
                        .font(.system(size: 11, weight: .semibold))
                        .monospacedDigit()
                }
                .foregroundStyle(Color.accentColor)
                .padding(.horizontal, 8)
                .padding(.vertical, 3.5)
                .background(Capsule().fill(Color.accentColor.opacity(0.12)))
            ) : nil
        ) {
            ZStack {
                RoundedRectangle(cornerRadius: 9)
                    .fill(Color(red: 0.06, green: 0.065, blue: 0.08))

                if let img = vm.currentPreviewImage {
                    Image(nsImage: img)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .padding(6)
                } else if let player = vm.player {
                    VideoPlayer(player: player)
                        .clipShape(RoundedRectangle(cornerRadius: 9))
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: vm.activeModelSpec.modality == .image ? "photo" : "film")
                            .font(.system(size: 32))
                            .foregroundStyle(.gray)
                        Text(L(
                            "生成的视频或高清图像将自动下载并在此预览",
                            "Generated videos or HD images will download automatically and preview here"
                        ))
                            .font(.callout)
                            .foregroundStyle(.gray)
                    }
                }
            }
            .frame(minHeight: 330)

            if let url = vm.currentMediaURL {
                HStack(spacing: 8) {
                    Image(systemName: vm.currentPreviewImage != nil ? "photo" : "film")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(url.lastPathComponent)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let el = curElapsed {
                        Text(formatDuration(el, compact: true))
                            .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1.5)
                            .background(Capsule().fill(Color.primary.opacity(0.06)))
                    }
                    Spacer()
                    if vm.player != nil {
                        Button(L("重播", "Replay")) {
                            vm.player?.seek(to: .zero)
                            vm.player?.play()
                        }
                        .controlSize(.small)
                    } else {
                        Button(L("用预览打开", "Open")) {
                            NSWorkspace.shared.open(url)
                        }
                        .controlSize(.small)
                    }
                    Button(L("在 Finder 中显示", "Show in Finder")) {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                    .controlSize(.small)
                }
            }
        }
    }

    var mediaHistoryCard: some View {
        let displayLocale = Locale(identifier: i18n.isEN ? "en_US" : "zh_CN")
        return WorkspaceCard(
            title: L("本地历史", "Local History"),
            subtitle: L("\(vm.localMediaItems.count) 个媒体文件", "\(vm.localMediaItems.count) media files"),
            icon: "clock.arrow.circlepath",
            trailing: AnyView(
                Button {
                    vm.refreshLocalMedia()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help(L("刷新本地列表", "Refresh local media list"))
            )
        ) {
            if vm.localMediaItems.isEmpty {
                Text(L("暂无本地生成记录", "No local outputs yet"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 290)
            } else {
                ScrollView {
                    VStack(spacing: 3) {
                        ForEach(vm.localMediaItems) { item in
                            let selected = vm.currentMediaURL == item.url
                            Button {
                                vm.selectMedia(item.url)
                            } label: {
                                HStack(spacing: 7) {
                                    Image(systemName: item.isImage ? "photo" : "film")
                                        .font(.caption)
                                        .foregroundStyle(selected ? Color.accentColor : Color.secondary)
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(item.filename)
                                            .font(.caption.weight(.medium))
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                        Text(item.createdAt.formatted(
                                            Date.FormatStyle(date: .abbreviated, time: .shortened, locale: displayLocale)
                                        ))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer(minLength: 4)
                                    if let el = item.elapsedSeconds, el > 0 {
                                        Text(formatDuration(el, compact: true))
                                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                            .foregroundStyle(selected ? Color.accentColor : Color.secondary)
                                            .padding(.horizontal, 5)
                                            .padding(.vertical, 2)
                                            .background(
                                                Capsule()
                                                    .fill(selected ? Color.accentColor.opacity(0.16) : Color.primary.opacity(0.06))
                                            )
                                    }
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 6)
                                .background(
                                    RoundedRectangle(cornerRadius: 6)
                                        .fill(selected ? Color.accentColor.opacity(0.15) : Color.clear)
                                )
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.trailing, 2)
                }
                .compatScrollIndicatorMargins(.vertical, 2)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .frame(height: 330)
            }
        }
    }

}
