import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers
import ObjectiveC.runtime

// MARK: - 多模型与多模态数据结构

enum ModelModality: String, Codable {
    case videoAudio = "video_audio"
    case video = "video"
    case image = "image"
    case llmVLM = "llm_vlm"
    case comfyUI = "comfyui"
}

struct TaskModeItem: Identifiable, Hashable {
    let id: String
    let titleZh: String
    let titleEn: String
    let subtitleZh: String
    let subtitleEn: String
    let icon: String

    var title: String { L(titleZh, titleEn) }
    var subtitle: String { L(subtitleZh, subtitleEn) }

    static func == (lhs: TaskModeItem, rhs: TaskModeItem) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

struct CustomHFModelDTO: Codable, Identifiable, Hashable {
    var id: String
    var name: String
    var repo_id: String
    var include_pattern: String?
    var modality: String?
    var pipeline_arch: String?
    var category: String?
    var approx_size_gb: Double?
    var is_custom: Bool?
}

struct OpenModelSpec: Identifiable, Hashable {
    let id: String
    let nameZh: String
    let nameEn: String
    let repoId: String
    let badgeZh: String
    let badgeEn: String
    let summaryZh: String
    let summaryEn: String
    let approxSizeGB: Double
    let icon: String
    let modality: ModelModality
    let defaultSteps: Double
    let modes: [TaskModeItem]
    let defaultFiles: [ModelFileItem]
    var includePattern: String = ""
    var isCustom: Bool = false

    var name: String { L(nameZh, nameEn) }
    var badge: String { L(badgeZh, badgeEn) }
    var summary: String { L(summaryZh, summaryEn) }

    static func == (lhs: OpenModelSpec, rhs: OpenModelSpec) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    static func fromCustomDTO(_ dto: CustomHFModelDTO) -> OpenModelSpec {
        let mod: ModelModality
        switch dto.modality {
        case "video_audio": mod = .videoAudio
        case "image": mod = .image
        default: mod = .llmVLM
        }
        let sizeGB = dto.approx_size_gb ?? 0.0
        let arch = (dto.pipeline_arch ?? "gguf").uppercased()
        let badgeZh = sizeGB > 0 ? String(format: "自定义 %@ · %.1f GB", arch, sizeGB) : "自定义 HF · \(arch)"
        let badgeEn = sizeGB > 0 ? String(format: "Custom %@ · %.1f GB", arch, sizeGB) : "Custom HF · \(arch)"
        let patDesc = (dto.include_pattern?.isEmpty == false) ? " (\(dto.include_pattern!))" : ""
        let summaryZh = dto.category ?? "自定义 HuggingFace 仓库 \(dto.repo_id)\(patDesc)"
        let summaryEn = "Custom HuggingFace repo \(dto.repo_id)\(patDesc)"
        let modes: [TaskModeItem]
        let iconName: String
        let defSteps: Double
        switch mod {
        case .videoAudio, .video:
            iconName = "film.stack"
            defSteps = 35
            modes = [
                TaskModeItem(id: "t2va", titleZh: "文生音视频", titleEn: "Text to Video+Audio", subtitleZh: "输入镜头与声音描述，同步生成视频与立体声音轨", subtitleEn: "Generate HD video and synchronized stereo audio from prompts", icon: "film"),
                TaskModeItem(id: "fl2va", titleZh: "首尾帧生视频", titleEn: "First/Last Frame to Video", subtitleZh: "上传首帧或尾帧参考图，精准控制起止画面与运镜过渡", subtitleEn: "Upload start or end keyframes to lock composition and camera transitions", icon: "photo.on.rectangle.angled")
            ]
        case .image:
            iconName = "sparkles.rectangle.stack"
            defSteps = 20
            modes = [
                TaskModeItem(id: "t2i", titleZh: "高清文生图", titleEn: "HD Text-to-Image", subtitleZh: "自定义 HuggingFace 图像生成管线", subtitleEn: "Custom HuggingFace image generation pipeline", icon: "photo.artframe"),
                TaskModeItem(id: "i2i_edit", titleZh: "图生图重绘", titleEn: "Image-to-Image Restyle", subtitleZh: "上传参考底图并调节重绘幅度，保留构图生成全新细节", subtitleEn: "Upload a reference image and adjust denoising strength to restyle", icon: "wand.and.rays")
            ]
        case .llmVLM, .comfyUI:
            iconName = "shippingbox.fill"
            defSteps = 20
            modes = [
                TaskModeItem(id: "chat", titleZh: "深度推理对话", titleEn: "Deep Reasoning Chat", subtitleZh: "自定义 HuggingFace 模型对话、长思维链推理与代码创作", subtitleEn: "Custom HuggingFace model chat, CoT reasoning & coding", icon: "bubble.left.and.bubble.right"),
                TaskModeItem(id: "vlm", titleZh: "图像视觉理解", titleEn: "Vision Understanding", subtitleZh: "拖入图片进行画面解析、OCR 识别或反推提示词", subtitleEn: "Drop images for scene analysis, OCR, or prompt reverse-engineering", icon: "eye"),
                TaskModeItem(id: "api", titleZh: "API 协议网关", titleEn: "API Protocol Gateway", subtitleZh: "提供 OpenAI (/v1) 与 Anthropic (/v1/messages) 双协议公网接口", subtitleEn: "Public dual-protocol gateway compatible with OpenAI (/v1) & Anthropic (/v1/messages)", icon: "network")
            ]
        }
        return OpenModelSpec(
            id: dto.id,
            nameZh: dto.name,
            nameEn: dto.name,
            repoId: dto.repo_id,
            badgeZh: badgeZh,
            badgeEn: badgeEn,
            summaryZh: summaryZh,
            summaryEn: summaryEn,
            approxSizeGB: sizeGB,
            icon: iconName,
            modality: mod,
            defaultSteps: defSteps,
            modes: modes,
            defaultFiles: [],
            includePattern: dto.include_pattern ?? "",
            isCustom: true
        )
    }
}

struct ModelFileItem: Codable, Identifiable, Hashable {
    var id: String { name }
    var name: String
    var desc: String?
    var exists: Bool
    var size_gb: Double

    var localizedDesc: String? {
        guard let raw = desc, !raw.isEmpty else { return nil }
        return localizeModelFileDesc(raw)
    }
}

let OPEN_MODEL_CATALOG: [OpenModelSpec] = [
    OpenModelSpec(
        id: "minimax_h3",
        nameZh: "MiniMax-H3 (33B)",
        nameEn: "MiniMax-H3 (33B)",
        repoId: "MiniMaxAI/MiniMax-H3",
        badgeZh: "BF16 原生 · 134 GB",
        badgeEn: "Native BF16 · 134 GB",
        summaryZh: "全模态 BF16 原生满血精度同步生成高清视频与 32kHz 立体声音效、人声对白及配乐",
        summaryEn: "Full-modality native BF16 generation of HD video with synchronized 32kHz stereo SFX, dialogue & music",
        approxSizeGB: 134.1,
        icon: "film.stack",
        modality: .videoAudio,
        defaultSteps: 35,
        modes: [
            TaskModeItem(
                id: "t2va",
                titleZh: "文生音视频",
                titleEn: "Text to Video+Audio",
                subtitleZh: "输入镜头与声音描述，同步生成视频与立体声音轨",
                subtitleEn: "Generate HD video and synchronized stereo audio from scene & sound prompts",
                icon: "film"
            ),
            TaskModeItem(
                id: "fl2va",
                titleZh: "首尾帧生视频",
                titleEn: "First/Last Frame to Video",
                subtitleZh: "上传首帧或尾帧参考图，精准控制起止画面与运镜过渡",
                subtitleEn: "Upload start or end keyframes to lock composition and camera transitions",
                icon: "photo.on.rectangle.angled"
            )
        ],
        defaultFiles: (1...13).map { i in
            ModelFileItem(
                name: String(format: "FL2VA/transformer/model-%05d-of-00013.safetensors", i),
                desc: "DiT 音视频主干 BF16 \(i)/13",
                exists: false,
                size_gb: 0
            )
        } + (1...14).map { i in
            ModelFileItem(
                name: String(format: "FL2VA/text_encoder/model-%05d-of-00014.safetensors", i),
                desc: "多模态编码器 BF16 \(i)/14",
                exists: false,
                size_gb: 0
            )
        } + [
            ModelFileItem(name: "FL2VA/video_vae/source/model.safetensors", desc: "3D 视频 VAE BF16 (9.7 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "FL2VA/audio_vae/model.safetensors", desc: "32kHz 立体声 VAE BF16 (0.6 GB)", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "comfyui_studio",
        nameZh: "ComfyUI 节点工作台",
        nameEn: "ComfyUI Node Studio",
        repoId: "comfyanonymous/ComfyUI",
        badgeZh: "节点视频创作 · 云盘软链持久化",
        badgeEn: "Node Video Studio · Drive Symlink Persist",
        summaryZh: "一键拉起云端 ComfyUI + Manager 全屏节点画布，自定义节点、视频权重与工作流自动软链持久化至 Google Drive",
        summaryEn: "1-click cloud ComfyUI + Manager fullscreen node canvas with Google Drive symlink persistence for custom nodes, weights & workflows",
        approxSizeGB: 0.0,
        icon: "point.3.connected.trianglepath.dotted",
        modality: .comfyUI,
        defaultSteps: 25,
        modes: [
            TaskModeItem(
                id: "comfy_canvas",
                titleZh: "全屏节点画布",
                titleEn: "Fullscreen Node Canvas",
                subtitleZh: "一键部署并跳转浏览器全屏节点画布，复用 G4 保活、折线监控与自动休眠",
                subtitleEn: "Deploy & open fullscreen canvas in browser with G4 keepalive, live telemetry & auto-sleep",
                icon: "macwindow.badge.plus"
            ),
            TaskModeItem(
                id: "comfy_assets",
                titleZh: "极速下权重与回传成片",
                titleEn: "Fast Weights & Pull Outputs",
                subtitleZh: "8 线程极速下载视频模型至云盘 models 目录，并将 output 成片一键回传 Mac",
                subtitleEn: "8-thread fast model download to Drive models folder & 1-click output pull to Mac",
                icon: "externaldrive.badge.icloud"
            )
        ],
        defaultFiles: []
    ),
    OpenModelSpec(
        id: "qwen38_27b",
        nameZh: "Qwen3.8-27B",
        nameEn: "Qwen3.8-27B",
        repoId: "Qwen/Qwen3.8-27B",
        badgeZh: "BF16 原生 · 51.8 GB",
        badgeEn: "Native BF16 · 51.8 GB",
        summaryZh: "Hugging Face 榜首开源多模态大模型（BF16 原生满血精度，支持最高 256K 上下文，需大显存实例）",
        summaryEn: "Top Hugging Face open multimodal LLM (full native BF16 precision, up to 256K context, high-VRAM GPU)",
        approxSizeGB: 51.8,
        icon: "brain",
        modality: .llmVLM,
        defaultSteps: 20,
        modes: [
            TaskModeItem(
                id: "chat",
                titleZh: "深度推理对话",
                titleEn: "Deep Reasoning Chat",
                subtitleZh: "支持长思维链推理、代码架构、复杂逻辑与分镜脚本创作",
                subtitleEn: "Long chain-of-thought reasoning, software architecture, logic & storyboard scripting",
                icon: "bubble.left.and.bubble.right"
            ),
            TaskModeItem(
                id: "vlm",
                titleZh: "图像视觉理解",
                titleEn: "Vision Understanding",
                subtitleZh: "拖入图片进行画面解析、OCR 识别或反推专业生图/生视频提示词",
                subtitleEn: "Drop images for scene analysis, OCR, or reverse-engineering image/video prompts",
                icon: "eye"
            ),
            TaskModeItem(
                id: "api",
                titleZh: "API 协议网关",
                titleEn: "API Protocol Gateway",
                subtitleZh: "提供 OpenAI (/v1) 与 Anthropic (/v1/messages) 双协议公网接口",
                subtitleEn: "Public dual-protocol gateway compatible with OpenAI (/v1) & Anthropic (/v1/messages)",
                icon: "network"
            )
        ],
        defaultFiles: (1...18).map { i in
            ModelFileItem(
                name: String(format: "model-%05d-of-00018.safetensors", i),
                desc: "权重分片 \(i)/18",
                exists: false,
                size_gb: 0
            )
        }
    ),
    OpenModelSpec(
        id: "qwen38_27b_gguf",
        nameZh: "Qwen3.8-27B-GGUF",
        nameEn: "Qwen3.8-27B-GGUF",
        repoId: "unsloth/Qwen3.8-27B-GGUF",
        badgeZh: "UD-Q4_K_M + MTP · 18.7 GB",
        badgeEn: "UD-Q4_K_M + MTP · 18.7 GB",
        summaryZh: "Unsloth 动态 4-bit 量化 + 原生 MTP 多词元投机解码加速，支持最高 256K 上下文与多模态视觉理解",
        summaryEn: "Unsloth dynamic 4-bit quantization + native MTP speculative decoding (~100 tok/s), up to 256K context & vision",
        approxSizeGB: 18.7,
        icon: "cpu",
        modality: .llmVLM,
        defaultSteps: 20,
        modes: [
            TaskModeItem(
                id: "chat",
                titleZh: "深度推理对话",
                titleEn: "Deep Reasoning Chat",
                subtitleZh: "UD-Q4_K_M + MTP 投机解码极速推理，支持长思维链、代码架构与超长上下文分析",
                subtitleEn: "Ultra-fast UD-Q4_K_M + MTP speculative decoding for CoT reasoning, coding & long-context analysis",
                icon: "bubble.left.and.bubble.right"
            ),
            TaskModeItem(
                id: "vlm",
                titleZh: "图像视觉理解",
                titleEn: "Vision Understanding",
                subtitleZh: "内置 mmproj-F16 视觉投影器，支持图片解析、OCR 与生图/生视频提示词反推",
                subtitleEn: "Built-in mmproj-F16 vision projector for image analysis, OCR & prompt reverse-engineering",
                icon: "eye"
            ),
            TaskModeItem(
                id: "api",
                titleZh: "API 协议网关",
                titleEn: "API Protocol Gateway",
                subtitleZh: "提供 OpenAI (/v1) 与 Anthropic (/v1/messages) 双协议公网接口",
                subtitleEn: "Public dual-protocol gateway compatible with OpenAI (/v1) & Anthropic (/v1/messages)",
                icon: "network"
            )
        ],
        defaultFiles: [
            ModelFileItem(name: "Qwen3.8-27B-UD-Q4_K_M.gguf", desc: "Unsloth UD-Q4_K_M 主模型 (16.5 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "mmproj-F16.gguf", desc: "多模态视觉投影器 (0.93 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", desc: "Unsloth MTP 多词元投机解码头 (1.31 GB)", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "qwen_image_21",
        nameZh: "Qwen-Image-2.1",
        nameEn: "Qwen-Image-2.1",
        repoId: "Qwen/Qwen-Image-2.1",
        badgeZh: "文生图与修图 · 30.9 GB",
        badgeEn: "Text-to-Image & Edit · 30.9 GB",
        summaryZh: "通义千问新一代旗舰图像生成与自然语言多图指令编辑模型",
        summaryEn: "Qwen flagship image generation & natural-language multi-image instruction editing model",
        approxSizeGB: 30.9,
        icon: "paintbrush.pointed",
        modality: .image,
        defaultSteps: 40,
        modes: [
            TaskModeItem(
                id: "t2i",
                titleZh: "高清文生图",
                titleEn: "HD Text-to-Image",
                subtitleZh: "复杂构图、精准中英文排版与高质感摄影/插画生成",
                subtitleEn: "Complex composition, accurate bilingual typography & high-fidelity photography/illustration",
                icon: "photo.artframe"
            ),
            TaskModeItem(
                id: "i2i_edit",
                titleZh: "多图指令修图",
                titleEn: "Multi-Image Instruction Edit",
                subtitleZh: "上传 1~2 张原图，用自然语言指令改局部、换背景或融合主体",
                subtitleEn: "Upload 1–2 source images to edit regions, swap backgrounds, or fuse subjects via natural language",
                icon: "wand.and.rays"
            )
        ],
        defaultFiles: [
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", desc: "DiT 1/2", exists: false, size_gb: 0),
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", desc: "DiT 2/2", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00001-of-00004.safetensors", desc: "编码器 1/4", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00002-of-00004.safetensors", desc: "编码器 2/4", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00003-of-00004.safetensors", desc: "编码器 3/4", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00004-of-00004.safetensors", desc: "编码器 4/4", exists: false, size_gb: 0),
            ModelFileItem(name: "vae/diffusion_pytorch_model.safetensors", desc: "图像 VAE", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "z_image_turbo",
        nameZh: "Z-Image-Turbo",
        nameEn: "Z-Image-Turbo",
        repoId: "Tongyi-MAI/Z-Image-Turbo",
        badgeZh: "8步极速生图 · 30.6 GB",
        badgeEn: "8-Step Turbo Image · 30.6 GB",
        summaryZh: "Hugging Face 高人气 8 步蒸馏极速写实摄影与中英文海报生图模型",
        summaryEn: "Popular Hugging Face 8-step distilled model for ultra-fast photorealism & bilingual poster design",
        approxSizeGB: 30.6,
        icon: "bolt.square",
        modality: .image,
        defaultSteps: 8,
        modes: [
            TaskModeItem(
                id: "t2i",
                titleZh: "8 步极速文生图",
                titleEn: "8-Step Fast Text-to-Image",
                subtitleZh: "仅需 8 步采样，秒级输出商业级人像摄影与中英文排版画面",
                subtitleEn: "Commercial-grade portrait photography & bilingual poster typography in just 8 steps",
                icon: "bolt"
            ),
            TaskModeItem(
                id: "i2i_edit",
                titleZh: "图生图风格重绘",
                titleEn: "Image-to-Image Restyle",
                subtitleZh: "上传参考底图并调节重绘幅度，保留构图生成全新细节",
                subtitleEn: "Upload a reference image and adjust denoising strength to restyle while preserving layout",
                icon: "square.on.square"
            )
        ],
        defaultFiles: [
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00001-of-00003.safetensors", desc: "DiT 1/3", exists: false, size_gb: 0),
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00002-of-00003.safetensors", desc: "DiT 2/3", exists: false, size_gb: 0),
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00003-of-00003.safetensors", desc: "DiT 3/3", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00001-of-00003.safetensors", desc: "编码器 1/3", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00002-of-00003.safetensors", desc: "编码器 2/3", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00003-of-00003.safetensors", desc: "编码器 3/3", exists: false, size_gb: 0),
            ModelFileItem(name: "vae/diffusion_pytorch_model.safetensors", desc: "图像 VAE", exists: false, size_gb: 0)
        ]
    )
]

struct ResolutionOption: Identifiable, Hashable {
    let id: String
    let labelZh: String
    let labelEn: String
    let width: Int
    let height: Int

    var label: String { L(labelZh, labelEn) }

    static func == (lhs: ResolutionOption, rhs: ResolutionOption) -> Bool {
        lhs.id == rhs.id && lhs.width == rhs.width && lhs.height == rhs.height
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(width)
        hasher.combine(height)
    }
}

let VIDEO_RESOLUTIONS: [ResolutionOption] = [
    ResolutionOption(id: "v_832_480", labelZh: "832 × 480 (16:9 横屏标准 · 推荐)", labelEn: "832 × 480 (16:9 Landscape · Recommended)", width: 832, height: 480),
    ResolutionOption(id: "v_480_832", labelZh: "480 × 832 (9:16 竖屏短视频)", labelEn: "480 × 832 (9:16 Vertical Short)", width: 480, height: 832),
    ResolutionOption(id: "v_1280_720", labelZh: "1280 × 720 (16:9 720P 高清)", labelEn: "1280 × 720 (16:9 720P HD)", width: 1280, height: 720),
    ResolutionOption(id: "v_720_1280", labelZh: "720 × 1280 (9:16 720P 竖屏)", labelEn: "720 × 1280 (9:16 720P Vertical)", width: 720, height: 1280),
    ResolutionOption(id: "v_640_640", labelZh: "640 × 640 (1:1 方形)", labelEn: "640 × 640 (1:1 Square)", width: 640, height: 640)
]

let IMAGE_RESOLUTIONS: [ResolutionOption] = [
    ResolutionOption(id: "i_1024_1024", labelZh: "1024 × 1024 (1:1 超清方图 · 推荐)", labelEn: "1024 × 1024 (1:1 Square HD · Recommended)", width: 1024, height: 1024),
    ResolutionOption(id: "i_1344_768", labelZh: "1344 × 768 (16:9 电影横幅)", labelEn: "1344 × 768 (16:9 Cinematic Widescreen)", width: 1344, height: 768),
    ResolutionOption(id: "i_768_1344", labelZh: "768 × 1344 (9:16 竖版海报)", labelEn: "768 × 1344 (9:16 Vertical Poster)", width: 768, height: 1344),
    ResolutionOption(id: "i_1536_1024", labelZh: "1536 × 1024 (3:2 单反摄影)", labelEn: "1536 × 1024 (3:2 DSLR Photo)", width: 1536, height: 1024)
]

struct PresetTemplate: Identifiable {
    let id: String
    let category: ModelModality
    let nameZh: String
    let nameEn: String
    let visualZh: String
    let visualEn: String
    let soundscape: String
    let music: String

    var name: String { L(nameZh, nameEn) }
    var visual: String { L(visualZh, visualEn) }
}

let PRESET_TEMPLATES: [PresetTemplate] = [
    PresetTemplate(
        id: "tpl_shinkai_train",
        category: .videoAudio,
        nameZh: "[动漫] 新海诚风 · 盛夏樱花铁道口",
        nameEn: "[Anime] Makoto Shinkai · Cherry Blossom Railway",
        visualZh: "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        visualEn: "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        soundscape: "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.",
        music: "Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings."
    ),
    PresetTemplate(
        id: "tpl_ufotable_duel",
        category: .videoAudio,
        nameZh: "[动漫] 飞碟社风 · 苍蓝雷光剑术对决",
        nameEn: "[Anime] Ufotable Sakuga · Azure Lightning Sword Duel",
        visualZh: "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        visualEn: "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        soundscape: "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, heavy stone debris crumbling, fierce battle cry.",
        music: "Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs."
    ),
    PresetTemplate(
        id: "tpl_iceland_fpv",
        category: .videoAudio,
        nameZh: "[写实] 冰岛巨型瀑布 FPV 穿越机航拍",
        nameEn: "[Realism] Iceland Waterfall FPV Drone Dive",
        visualZh: "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt columns under golden hour sunlight, mist rising into a rainbow.",
        visualEn: "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt columns under golden hour sunlight, mist rising into a rainbow.",
        soundscape: "Roaring waterfall rushing water, wind whooshing past the camera, distant seabirds calling.",
        music: "Epic cinematic acoustic guitar and soaring cello."
    ),
    PresetTemplate(
        id: "tpl_cyber_poster",
        category: .image,
        nameZh: "[生图] 东方赛博霓虹商业海报（含中文排版）",
        nameEn: "[Image] Cyberpunk Neon Bund Poster (Chinese Title)",
        visualZh: "超清商业电影海报摄影，雨夜霓虹闪烁的未来上海外滩街道，一位身着机能风银色夹克的少女回眸凝视镜头，景深虚化光斑，顶部醒目发光现代中文字体排版写着“未来纪元 2026”，极致光影细节，8K画质。",
        visualEn: "Ultra-HD commercial sci-fi movie poster photography, rain-slicked cyberpunk Shanghai Bund street glowing with neon reflections at night, a young woman in a futuristic silver techwear jacket looking back at the camera, shallow depth of field bokeh flares, bold glowing modern title typography at the top reading \"FUTURE ERA 2026\" (未来纪元 2026), intricate lighting details, 8K.",
        soundscape: "",
        music: ""
    ),
    PresetTemplate(
        id: "tpl_film_cafe",
        category: .image,
        nameZh: "[生图] 胶片质感午后咖啡馆肖像",
        nameEn: "[Image] Hasselblad Film Afternoon Tokyo Cafe",
        visualZh: "Hasselblad medium format film photography, soft golden afternoon sunlight streaming through wooden blinds onto a cozy minimalist Tokyo cafe table, a ceramic pour-over coffee dripper with rising steam beside an open sketchbook, natural film grain, ultra-detailed textures.",
        visualEn: "Hasselblad medium format film photography, soft golden afternoon sunlight streaming through wooden blinds onto a cozy minimalist Tokyo cafe table, a ceramic pour-over coffee dripper with rising steam beside an open sketchbook, natural film grain, ultra-detailed textures.",
        soundscape: "",
        music: ""
    ),
    PresetTemplate(
        id: "tpl_llm_storyboard",
        category: .llmVLM,
        nameZh: "[推理] 为 AI 视频导演生成 3 组新海诚风分镜提示词",
        nameEn: "[Reasoning] Generate 3 Shinkai-Style AI Video Storyboard Prompts",
        visualZh: "请帮我构思 3 组极具画面感的新海诚风格动画短片分镜，每组包含：1) 英文 Visual & Camera 镜头提示词；2) Soundscape 环境音描述；3) Background Music 配乐描述。",
        visualEn: "Please design 3 vivid Makoto Shinkai-style anime short film storyboard shots. For each shot, provide: 1) Visual & Camera prompt; 2) Soundscape & SFX prompt; 3) Background Music prompt.",
        soundscape: "",
        music: ""
    ),
    PresetTemplate(
        id: "tpl_vlm_reverse",
        category: .llmVLM,
        nameZh: "[视觉] 深度解析图片构图并反推生图/生视频 Prompt",
        nameEn: "[Vision] Analyze Image Composition & Reverse-Engineer Prompt",
        visualZh: "请仔细观察这张图片的主体、光影色调、镜头景别与艺术风格，先做简要中文分析，然后反推出一段高质量的英文生图/生视频 Prompt。",
        visualEn: "Carefully analyze the subject, lighting, color palette, camera framing, and artistic style of this image. First give a concise visual breakdown, then write a high-quality English image/video generation prompt.",
        soundscape: "",
        music: ""
    )
]

// MARK: - 网络协议结构

struct LoadedModelSummary: Codable, Identifiable, Hashable {
    var id: String
    var name: String
    var mode: String?
    var vram_gb: Double?
    var context_length: Int?

    var localizedName: String {
        if let spec = OPEN_MODEL_CATALOG.first(where: { $0.id == id }) {
            return spec.name
        }
        return localizeServerText(name)
    }
}

struct ColabStatusResponse: Codable {
    var ok: Bool?
    var cuda_available: Bool?
    var gpu_name: String?
    var vram_free_gb: Double?
    var vram_total_gb: Double?
    var vram_used_gb: Double?
    var model_loaded: Bool?
    var any_model_loaded: Bool?
    var selected_model_id: String?
    var loaded_model_id: String?
    var loaded_model_ids: [String]?
    var loaded_models_detail: [LoadedModelSummary]?
    var current_model: String?
    var message: String?
    var context_length: Int?
    var loaded_context_length: Int?

    var files: [ModelFileItem]?
    var downloaded_count: Int?
    var total_count: Int?
    var all_downloaded: Bool?
    var total_size_gb: Double?
    var drive_persisted: Bool?
    var bf16_shards_count: Int?
    var is_downloading: Bool?
    var is_loading_vram: Bool?
    var task_progress: Double?
    var task_message: String?

    var uptime_seconds: Int?
    var cu_burn_rate_per_hour: Double?
    var session_cu_used: Double?
    var idle_timeout_minutes: Int?
    var idle_remaining_seconds: Int?
    var api_key_enabled: Bool?
    var api_key_configured: Bool?
    var api_request_count: Int?
    var console_lines: [String]?

    var ram_total_gb: Double?
    var ram_used_gb: Double?
    var ram_free_gb: Double?
    var disk_total_gb: Double?
    var disk_used_gb: Double?
    var disk_free_gb: Double?
    var shm_total_gb: Double?
    var shm_used_gb: Double?
    var shm_free_gb: Double?
    var drive_total_gb: Double?
    var drive_used_gb: Double?
    var drive_free_gb: Double?
    var comfyui: ComfyRemoteStatus?
    var custom_models: [CustomHFModelDTO]?
    var registered_model: CustomHFModelDTO?
    var action_ok: Bool?
    var action_message: String?
}

struct ComfyOutputMediaItem: Codable, Identifiable, Hashable {
    var id: String { rel_path }
    var rel_path: String
    var filename: String
    var media_type: String?
    var size_mb: Double?
    var mtime: Int?
}

struct ComfyPresetPackItem: Codable, Identifiable, Hashable {
    var id: String
    var title: String
    var badge: String
    var scenario: String
    var approx_size_gb: Double
    var workflows: [String]
    var installed_models: Int
    var total_models: Int
    var workflows_installed: Bool
    var ready: Bool

    var localizedTitle: String {
        guard LanguageManager.shared.isEN else { return title }
        switch id {
        case "wan22_i2v_flf2v":
            return "Wan2.2 14B · Image-to-Video + First/Last Frame Dual Workflows"
        case "minimax_h3_i2v_flf2v":
            return "MiniMax-H3 Turbo · Image-to-Video+Audio + First/Last Frame"
        case "ltx23_flf2v":
            return "LTX-2.3 (22B) · Cinematic Camera Motion First/Last Frame + Audio"
        case "hunyuan15_i2v_1080p":
            return "HunyuanVideo 1.5 · Single-Image Direct 1080P Super-Res Workflow"
        default:
            return localizeServerText(title)
        }
    }

    var localizedBadge: String {
        guard LanguageManager.shared.isEN else { return badge }
        switch id {
        case "wan22_i2v_flf2v":
            return "Beginner Pick · 4-Step Fast"
        case "minimax_h3_i2v_flf2v":
            return "Stereo Audio · 8-Step Turbo"
        case "ltx23_flf2v":
            return "Cinematic Camera · Only 2 Weights"
        case "hunyuan15_i2v_1080p":
            return "720P + 1080P Dual Output"
        default:
            return localizeServerText(badge)
        }
    }

    var localizedScenario: String {
        guard LanguageManager.shared.isEN else { return scenario }
        switch id {
        case "wan22_i2v_flf2v":
            return "Includes 'Single Image to Video' & 'First/Last Frame Smooth Transition' templates (~25s on G4, 100% built-in nodes, zero missing-node errors)"
        case "minimax_h3_i2v_flf2v":
            return "Includes single-image and first/last-frame templates generating HD video with synchronized stereo SFX, dialogue & music in one pass"
        case "ltx23_flf2v":
            return "Optimized for dramatic camera moves between start & end keyframes (e.g. aerial crane dive to low-angle close-up + synthesized soundtrack)"
        case "hunyuan15_i2v_1080p":
            return "Latent super-resolution pipeline enabled by default — upload 1 image to render both 720P base video and 1920×1080 Full HD output"
        default:
            return localizeServerText(scenario)
        }
    }

    static func localizedWorkflowLabel(_ rawName: String) -> String {
        let stripped = rawName.replacingOccurrences(of: ".json", with: "")
        guard LanguageManager.shared.isEN else { return stripped }
        let map: [String: String] = [
            "01_Wan2.2_极速单图生视频(4步傻瓜版)": "01_Wan2.2_Fast_Single_I2V_4Step",
            "02_Wan2.2_首尾帧丝滑过渡(双图4步极速)": "02_Wan2.2_FirstLastFrame_4Step",
            "03_MiniMax-H3_极速图生有声视频(单图带立体声)": "03_MiniMax-H3_Fast_I2VA_Stereo",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声)": "04_MiniMax-H3_FirstLastFrame_Stereo",
            "05_LTX-2.3_首尾帧电影运镜(双图极速带音效)": "05_LTX-2.3_Cinematic_FLF2V_Audio",
            "06_Hunyuan1.5_单图直出1080P超清视频": "06_Hunyuan1.5_Direct_1080P_I2V"
        ]
        return map[stripped] ?? stripped
    }
}

let DEFAULT_COMFY_PRESET_PACKS: [ComfyPresetPackItem] = [
    ComfyPresetPackItem(
        id: "wan22_i2v_flf2v",
        title: "万相 Wan2.2 14B · 图生视频 + 首尾帧双工作流",
        badge: "新手首选 · 4步极速",
        scenario: "含「单图动起来」与「首尾双图丝滑过渡」2 套中文画布，G4 约 25 秒出片，纯官方内置节点零红框报错",
        approx_size_gb: 35.4,
        workflows: [
            "01_Wan2.2_极速单图生视频(4步傻瓜版).json",
            "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json"
        ],
        installed_models: 0,
        total_models: 6,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_i2v_flf2v",
        title: "MiniMax-H3 Turbo · 图生有声视频 + 首尾帧双工作流",
        badge: "自带立体声音效 · 8步加速",
        scenario: "含「单图生有声视频」与「首尾双图有声过渡」2 套极简画布，一次生成高清动态画面 + 同步环境音效/配乐/对白",
        approx_size_gb: 39.1,
        workflows: [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json"
        ],
        installed_models: 0,
        total_models: 5,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "ltx23_flf2v",
        title: "LTX-2.3 (22B) · 首尾帧电影级运镜有声工作流",
        badge: "电影推拉摇移 · 仅 2 个权重",
        scenario: "专为首尾关键帧大幅度镜头运动优化（如高空俯拍平滑推镜至低角度特写 + 自动合成配乐音轨）",
        approx_size_gb: 36.3,
        workflows: [
            "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json"
        ],
        installed_models: 0,
        total_models: 2,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "hunyuan15_i2v_1080p",
        title: "混元 HunyuanVideo 1.5 · 单图直出 1080P 超分工作流",
        badge: "720P + 1080P 双路超清",
        scenario: "已默认激活潜空间超分管线，传 1 张图一键自动渲染 720P 原片与 1920×1080 超高清成片",
        approx_size_gb: 43.5,
        workflows: [
            "06_Hunyuan1.5_单图直出1080P超清视频.json"
        ],
        installed_models: 0,
        total_models: 7,
        workflows_installed: false,
        ready: false
    )
]

struct ComfyRemoteStatus: Codable {
    var is_running: Bool?
    var is_starting: Bool?
    var is_syncing_drive: Bool?
    var is_downloading_weight: Bool?
    var installing_pack_id: String?
    var progress: Double?
    var status_message: String?
    var public_url: String?
    var local_port: Int?
    var queue_running: Int?
    var queue_pending: Int?
    var persistence_mode: String?
    var persistence_label: String?
    var drive_persisted: Bool?
    var persist_root: String?
    var custom_nodes_count: Int?
    var models_count: Int?
    var workflows_count: Int?
    var last_sync_ts: Int?
    var recent_outputs: [ComfyOutputMediaItem]?
    var preset_packs: [ComfyPresetPackItem]?
}

struct CompletedMediaItem: Codable, Hashable {
    var index: Int?
    var seed: Int?
    var filename: String
    var video_url: String
    var media_type: String?
    var elapsed: Double?
}

struct LiveTelemetrySnapshot: Codable {
    var cuda_available: Bool?
    var gpu_name: String?
    var vram_free_gb: Double?
    var vram_total_gb: Double?
    var vram_used_gb: Double?
    var cu_burn_rate_per_hour: Double?
    var ram_total_gb: Double?
    var ram_used_gb: Double?
    var ram_free_gb: Double?
    var disk_total_gb: Double?
    var disk_used_gb: Double?
    var disk_free_gb: Double?
    var shm_total_gb: Double?
    var shm_used_gb: Double?
    var shm_free_gb: Double?
    var drive_total_gb: Double?
    var drive_used_gb: Double?
    var drive_free_gb: Double?
}

struct ColabJobResponse: Codable {
    var job_id: String
    var model_id: String?
    var status: String
    var progress: Double
    var message: String
    var seed: Int?
    var elapsed: Double?
    var filename: String?
    var video_url: String?
    var media_type: String?
    var text_output: String?
    var thinking_output: String?
    var batch_count: Int?
    var completed_count: Int?
    var completed_videos: [CompletedMediaItem]?
    var console_lines: [String]?
    var telemetry: LiveTelemetrySnapshot?
}

struct BridgeAuthStatusResponse: Codable {
    var ok: Bool?
    var logged_in: Bool?
    var ws_connected: Bool?
    var email: String?
    var paid_balance: Double?
    var burn_rate_hourly: Double?
    var active_assignments: Int?
    var message: String?
    var warning: String?
    var error: String?
}

struct LocalMediaItem: Identifiable, Hashable {
    let id = UUID()
    let url: URL
    let filename: String
    let isImage: Bool
    let createdAt: Date
    let elapsedSeconds: Double?
}

func formatDuration(_ sec: Double, compact: Bool = false) -> String {
    let s = max(0.0, sec)
    if s < 60.0 {
        if compact { return String(format: "%.1fs", s) }
        return LanguageManager.shared.isEN
            ? String(format: "%.1f s", s)
            : String(format: "%.1f 秒", s)
    }
    let mins = Int(s) / 60
    let rem = s - Double(mins * 60)
    if compact { return String(format: "%dm%.0fs", mins, rem) }
    return LanguageManager.shared.isEN
        ? String(format: "%dm %.1fs", mins, rem)
        : String(format: "%d分%.1f秒", mins, rem)
}

struct HFTrendingModelItem: Identifiable, Codable, Hashable {
    var id: String { repo_id }
    let repo_id: String
    let name: String
    let pipeline_tag: String
    let modality: String
    let is_gguf: Bool
    let is_gated: Bool
    let suggested_pattern: String
    let likes: Int
    let downloads: Int
    var param_count: Int? = nil
    var param_label: String? = nil
    var est_vram_gb: Double? = nil
    var deploy_status: String? = nil
    var deploy_label_zh: String? = nil
    var deploy_label_en: String? = nil

    var localizedDeployLabel: String {
        let rawZh = (deploy_label_zh?.isEmpty == false) ? deploy_label_zh! : "显存待探测"
        let rawEn = (deploy_label_en?.isEmpty == false) ? deploy_label_en! : "VRAM TBD"
        return stripEmojiCharacters(L(rawZh, rawEn))
    }

    var deployBadgeColor: Color {
        switch deploy_status ?? "unknown" {
        case "great", "good":
            return .green
        case "a100":
            return Color(red: 0.85, green: 0.62, blue: 0.05)
        case "heavy":
            return .orange
        case "too_large":
            return .red
        default:
            return .secondary
        }
    }
}


struct HFTrendingAPIResponse: Codable {
    var ok: Bool?
    var category: String?
    var models: [HFTrendingModelItem]?
}

struct HFModelWeightFileItem: Identifiable, Codable, Hashable {
    var id: String { filename }
    let filename: String
    let size_gb: Double
    let fit_status: String
    let fit_zh: String
    let fit_en: String

    var localizedFit: String { stripEmojiCharacters(L(fit_zh, fit_en)) }

    var fitColor: Color {
        switch fit_status {
        case "great", "good": return .green
        case "a100": return Color(red: 0.85, green: 0.62, blue: 0.05)
        case "heavy": return .orange
        case "too_large": return .red
        default: return .secondary
        }
    }
}

struct HFModelDetailInfo: Identifiable, Codable, Hashable {
    var id: String { repo_id }
    var ok: Bool?
    var repo_id: String
    var name: String
    var author: String
    var pipeline_tag: String
    var modality: String
    var is_gguf: Bool
    var is_gated: Bool
    var license: String
    var base_model: String
    var architecture: String
    var context_length: Int
    var languages: [String]
    var likes: Int
    var downloads: Int
    var last_modified: String
    var used_storage_gb: Double
    var param_count: Int
    var param_label: String
    var est_vram_gb: Double
    var deploy_status: String
    var deploy_label_zh: String
    var deploy_label_en: String
    var weight_files: [HFModelWeightFileItem]
    var readme_excerpt: String

    var localizedDeployLabel: String {
        stripEmojiCharacters(L(deploy_label_zh.isEmpty ? "显存待探测" : deploy_label_zh, deploy_label_en.isEmpty ? "VRAM TBD" : deploy_label_en))
    }

    var deployBadgeColor: Color {
        switch deploy_status {
        case "great", "good": return .green
        case "a100": return Color(red: 0.85, green: 0.62, blue: 0.05)
        case "heavy": return .orange
        case "too_large": return .red
        default: return .secondary
        }
    }
}

struct ChatTurnItem: Identifiable, Hashable {
    let id = UUID()
    var role: String      // "user" or "assistant"
    var content: String
    var thinking: String = ""
    var hasImage: Bool = false
}


// MARK: - 界面色彩主题（左中右三栏高辨识度分界）

struct StudioPalette {
    let colorScheme: ColorScheme

    // 左侧控制栏：冷灰蓝底色
    var sidebarBackground: Color {
        colorScheme == .dark
            ? Color(red: 0.110, green: 0.120, blue: 0.145)
            : Color(red: 0.912, green: 0.928, blue: 0.955)
    }

    // 左侧卡片底色
    var sidebarCardFill: Color {
        colorScheme == .dark
            ? Color(red: 0.155, green: 0.168, blue: 0.200)
            : Color(red: 0.985, green: 0.990, blue: 1.000)
    }

    // 中间主工作区画布底色
    var workspaceCanvas: Color {
        colorScheme == .dark
            ? Color(red: 0.072, green: 0.078, blue: 0.094)
            : Color(red: 0.974, green: 0.978, blue: 0.986)
    }

    // 中间工作区卡片底色
    var workspaceCardFill: Color {
        colorScheme == .dark
            ? Color(red: 0.125, green: 0.135, blue: 0.162)
            : Color.white
    }

    // 右侧运行日志面板顶栏底色
    var logHeaderBackground: Color {
        colorScheme == .dark
            ? Color(red: 0.102, green: 0.112, blue: 0.136)
            : Color(red: 0.924, green: 0.934, blue: 0.954)
    }

    // 右侧运行日志面板背景色
    var logPanelBackground: Color {
        colorScheme == .dark
            ? Color(red: 0.088, green: 0.096, blue: 0.118)
            : Color(red: 0.938, green: 0.945, blue: 0.960)
    }

    // 右侧深色终端视口底色
    var logTerminalFill: Color {
        Color(red: 0.072, green: 0.082, blue: 0.104)
    }

    // 左中右面板分界线颜色
    var splitDivider: Color {
        colorScheme == .dark
            ? Color.white.opacity(0.14)
            : Color(red: 0.76, green: 0.80, blue: 0.86)
    }

    // 卡片边框色
    var cardBorder: Color {
        colorScheme == .dark
            ? Color.white.opacity(0.09)
            : Color(red: 0.82, green: 0.85, blue: 0.90)
    }

    // 输入框底色
    var inputFill: Color {
        colorScheme == .dark
            ? Color(red: 0.090, green: 0.098, blue: 0.118)
            : Color(red: 0.985, green: 0.988, blue: 0.994)
    }
}

// MARK: - 跨环境动态路径与自包含运行支撑

enum LocalRuntimePaths {
    /// 若应用位于源码仓库目录 (<repo>/EasyColab.app)，优先读取仓库内文件以便开发热更新；
    /// 若被独立安装至 /Applications，自动回退使用 EasyColab.app/Contents/Resources 内置资源。
    static var repoRootIfPresent: URL? {
        let fm = FileManager.default
        let bundleParent = Bundle.main.bundleURL.deletingLastPathComponent()
        if fm.fileExists(atPath: bundleParent.appendingPathComponent("app.py").path),
           fm.fileExists(atPath: bundleParent.appendingPathComponent("colab_bridge.py").path) {
            return bundleParent
        }
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
        if fm.fileExists(atPath: cwd.appendingPathComponent("app.py").path),
           fm.fileExists(atPath: cwd.appendingPathComponent("colab_bridge.py").path) {
            return cwd
        }
        return nil
    }

    static func resolveBundledOrRepoFile(_ filename: String) -> String {
        let fm = FileManager.default
        if let repo = repoRootIfPresent {
            let p = repo.appendingPathComponent(filename).path
            if fm.fileExists(atPath: p) { return p }
        }
        if let res = Bundle.main.resourceURL {
            let p = res.appendingPathComponent(filename).path
            if fm.fileExists(atPath: p) { return p }
        }
        return filename
    }

    static var appSupportDir: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = base.appendingPathComponent("EasyColab", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static var managedVenvDir: URL {
        appSupportDir.appendingPathComponent("venv", isDirectory: true)
    }

    static func findReadyBridgePython() -> String? {
        let fm = FileManager.default
        var candidates: [String] = []
        if let repo = repoRootIfPresent {
            candidates.append(repo.appendingPathComponent(".venv/bin/python").path)
            candidates.append(repo.appendingPathComponent("venv/bin/python").path)
        }
        candidates.append(managedVenvDir.appendingPathComponent("bin/python").path)
        for py in candidates where fm.isExecutableFile(atPath: py) {
            let check = Process()
            check.executableURL = URL(fileURLWithPath: py)
            check.arguments = ["-c", "import colab_cli, google.auth, websocket"]
            check.standardOutput = Pipe()
            check.standardError = Pipe()
            if (try? check.run()) != nil {
                check.waitUntilExit()
                if check.terminationStatus == 0 {
                    return py
                }
            }
        }
        return nil
    }

    static func findSystemPython3() -> String? {
        let fm = FileManager.default
        let candidates = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
            "/usr/bin/python3"
        ]
        return candidates.first { fm.isExecutableFile(atPath: $0) }
    }
}

