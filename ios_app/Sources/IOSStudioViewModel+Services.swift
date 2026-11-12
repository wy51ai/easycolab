import SwiftUI
import UIKit
import AuthenticationServices
import AVKit
import Photos

// MARK: - iOS / iPadOS Studio ViewModel
// MARK: - IOSStudioViewModel Extension: Custom HF Models, ComfyUI & Photos Gallery

extension IOSStudioViewModel {
    // MARK: - Custom HuggingFace Model Registry

    var customModelsStorageKey: String { "easycolab.ios.custom_models.v1" }

    func loadSavedCustomModels() {
        guard let data = UserDefaults.standard.data(forKey: customModelsStorageKey),
              let dtos = try? JSONDecoder().decode([CustomHFModelDTO].self, from: data) else {
            return
        }
        customModels = dtos.map { OpenModelSpec.fromCustomDTO($0) }
    }

    func saveCustomModelsToDisk(_ specs: [OpenModelSpec]) {
        let dtos: [CustomHFModelDTO] = specs.map { sp in
            let modStr: String
            switch sp.modality {
            case .videoAudio, .video: modStr = "video_audio"
            case .image: modStr = "image"
            case .llmVLM, .comfyUI: modStr = "llm_vlm"
            }
            return CustomHFModelDTO(
                id: sp.id,
                name: sp.nameZh,
                repo_id: sp.repoId,
                modality: modStr,
                category: sp.summaryZh,
                approx_size_gb: sp.approxSizeGB,
                include_pattern: sp.includePattern,
                pipeline_arch: sp.badgeZh.contains("GGUF") ? "gguf" : "transformers",
                files_count: sp.defaultFiles.count,
                is_custom: true
            )
        }
        if let data = try? JSONEncoder().encode(dtos) {
            UserDefaults.standard.set(data, forKey: customModelsStorageKey)
        }
    }

    func mergeRemoteCustomModels(_ dtos: [CustomHFModelDTO]) {
        var merged = customModels
        for dto in dtos {
            let spec = OpenModelSpec.fromCustomDTO(dto)
            if let idx = merged.firstIndex(where: { $0.id == spec.id || ($0.repoId.lowercased() == spec.repoId.lowercased() && $0.includePattern == spec.includePattern) }) {
                merged[idx] = spec
            } else {
                merged.append(spec)
            }
        }
        if merged != customModels {
            customModels = merged
            saveCustomModelsToDisk(merged)
        }
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
            if let url = comps?.url {
                var req = URLRequest(url: url)
                req.timeoutInterval = 12
                if let (data, resp) = try? await URLSession.shared.data(for: req),
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
                req.setValue("EasyColab-iOS/2.0", forHTTPHeaderField: "User-Agent")
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
        var cleanId = repoId.trimmingCharacters(in: .whitespacesAndNewlines)
        if let range = cleanId.range(of: "huggingface.co/") {
            cleanId = String(cleanId[range.upperBound...])
        }
        let parts = cleanId.split(separator: "/").map(String.init)
        guard parts.count >= 2 else { return }
        cleanId = "\(parts[0])/\(parts[1])"
        showHFModelDetailSheet = true
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
            if let url = comps?.url {
                var req = URLRequest(url: url)
                req.timeoutInterval = 14
                if let (data, resp) = try? await URLSession.shared.data(for: req),
                   (resp as? HTTPURLResponse)?.statusCode == 200,
                   let dec = try? JSONDecoder().decode(HFModelDetailInfo.self, from: data),
                   dec.ok == true {
                    activeHFModelDetail = dec
                    return
                }
            }
        }

        var metaObj: [String: Any]? = nil
        for host in ["https://huggingface.co", "https://hf-mirror.com"] {
            guard let url = URL(string: "\(host)/api/models/\(cleanId)?blobs=true") else { continue }
            var req = URLRequest(url: url)
            req.timeoutInterval = 12
            req.setValue("EasyColab-iOS/2.0", forHTTPHeaderField: "User-Agent")
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
        if lic.isEmpty { lic = L("未标注", "Unspecified") }

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
            rReq.setValue("EasyColab-iOS/2.0", forHTTPHeaderField: "User-Agent")
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
        let rawRepo = customRepoIdInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawRepo.isEmpty else {
            customModelBannerMessage = L("请输入 HuggingFace 仓库地址（例如 owner/repo）", "Please enter a HuggingFace Repo ID (e.g. owner/repo)")
            return
        }
        let pat = customIncludePatternInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let customName = customDisplayNameInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let hint = customModalityHint

        isRegisteringCustomModel = true
        customModelBannerMessage = L("正在解析 HuggingFace 仓库结构与权重分片...", "Inspecting HuggingFace repository & weight shards...")
        defer { isRegisteringCustomModel = false }

        let base = normalizedBaseURL
        if isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/custom_model/register") {
            do {
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.timeoutInterval = 35
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                req.httpBody = try JSONSerialization.data(withJSONObject: [
                    "repo_id": rawRepo,
                    "include_pattern": pat,
                    "custom_name": customName,
                    "modality_hint": hint,
                    "hf_token": hfToken,
                    "auto_download": autoDownload
                ])
                let (data, resp) = try await URLSession.shared.data(for: req)
                if (resp as? HTTPURLResponse)?.statusCode == 200,
                   let decoded = try? JSONDecoder().decode(ColabStatusResponse.self, from: data) {
                    applyStatusResponse(decoded)
                    if decoded.action_ok == true {
                        if let reg = decoded.registered_model {
                            let spec = OpenModelSpec.fromCustomDTO(reg)
                            mergeRemoteCustomModels([reg])
                            switchSelectedModel(spec)
                        }
                        let msg = localizeServerText(decoded.action_message ?? L("已成功添加自定义模型！", "Custom model registered!"))
                        customModelBannerMessage = msg
                        statusMessage = msg
                        showToast(L("已导入自定义模型", "Custom Model Imported"))
                        customRepoIdInput = ""
                        customIncludePatternInput = ""
                        customDisplayNameInput = ""
                        showAddCustomModelSheet = false
                        if autoDownload {
                            await startRemoteDownloadWeights()
                        }
                        return
                    } else {
                        let srvMsg = decoded.action_message ?? ""
                        if !srvMsg.contains("all_rfilenames") {
                            customModelBannerMessage = localizeServerText(srvMsg.isEmpty ? L("仓库解析失败，请检查 Repo ID 是否正确", "Failed to inspect repo, please check Repo ID") : srvMsg)
                            return
                        }
                    }
                }
            } catch {
                // 网络抖动或远端异常时回退至本地模型中心注册
            }
        }

        // 离线或尚未连接云端时：直接通过 HuggingFace 公开 API 校验并保存至本地模型中心，连接后可一键下载
        var cleanRepo = rawRepo
        if let range = cleanRepo.range(of: "huggingface.co/") {
            cleanRepo = String(cleanRepo[range.upperBound...])
        }
        let parts = cleanRepo.split(separator: "/").map(String.init)
        guard parts.count >= 2 else {
            customModelBannerMessage = L("Repo ID 格式应为 owner/repo（例如 unsloth/Qwen3.8-27B-GGUF）", "Repo ID must be owner/repo")
            return
        }
        let normRepo = "\(parts[0])/\(parts[1])"
        let display = customName.isEmpty ? parts[1] : customName
        let modStr: String
        if hint != "auto" {
            modStr = hint
        } else if normRepo.lowercased().contains("image") || normRepo.lowercased().contains("flux") || normRepo.lowercased().contains("sdxl") {
            modStr = "image"
        } else if normRepo.lowercased().contains("video") || normRepo.lowercased().contains("wan") || normRepo.lowercased().contains("ltx") {
            modStr = "video_audio"
        } else {
            modStr = "llm_vlm"
        }
        let archStr = (normRepo.lowercased().contains("gguf") || pat.lowercased().contains(".gguf")) ? "gguf" : "transformers"
        let localId = "custom_" + normRepo.lowercased().replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "-", with: "_").prefix(28)
        let dto = CustomHFModelDTO(
            id: String(localId),
            name: display,
            repo_id: normRepo,
            modality: modStr,
            category: "自定义 HuggingFace 仓库 · \(normRepo)",
            approx_size_gb: 0.0,
            include_pattern: pat,
            pipeline_arch: archStr,
            files_count: 1,
            is_custom: true
        )
        let spec = OpenModelSpec.fromCustomDTO(dto)
        mergeRemoteCustomModels([dto])
        switchSelectedModel(spec)
        customRepoIdInput = ""
        customIncludePatternInput = ""
        customDisplayNameInput = ""
        customModelBannerMessage = L("已添加 \(display) 至模型中心（连接云端后即可一键下载权重）", "Added \(display) to Model Center (connect to cloud to download weights)")
        showToast(L("已添加 \(display)", "Added \(display)"))
        showAddCustomModelSheet = false
    }

    func removeCustomHFModel(_ spec: OpenModelSpec) async {
        customModels.removeAll { $0.id == spec.id }
        saveCustomModelsToDisk(customModels)
        if selectedModelId == spec.id {
            switchSelectedModel(OPEN_MODEL_CATALOG[0])
        }
        let base = normalizedBaseURL
        if isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/custom_model/remove") {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try? JSONSerialization.data(withJSONObject: ["model_id": spec.id, "delete_files": false])
            if let (data, _) = try? await URLSession.shared.data(for: req),
               let dec = try? JSONDecoder().decode(ColabStatusResponse.self, from: data) {
                applyStatusResponse(dec)
            }
        }
        showToast(L("已移除 \(spec.shortTitle)", "Removed \(spec.shortTitle)"))
    }

    func checkColabStatus(silent: Bool = false) async {
        let base = normalizedBaseURL
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/status?model_id=\(selectedModelId)") else {
            if !silent {
                statusMessage = L("请先绑定账号一键唤醒，或填入 gradio.live 服务地址。", "Bind Google account to wake G4, or enter a gradio.live URL.")
            }
            isConnected = false
            return
        }
        if !silent {
            statusMessage = L("正在连接并同步云端 GPU 状态...", "Connecting & syncing cloud GPU status...")
        }
        do {
            var req = URLRequest(url: url)
            req.timeoutInterval = isGenerating ? 28 : 20
            let (data, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            guard code == 200 else {
                consecutiveStatusFailures += 1
                if !silent || consecutiveStatusFailures >= 6 {
                    resetStaleRemoteRuntimeState(reason: L("云端连接不可用 (HTTP \(code))，实例可能已休眠或释放。", "Cloud runtime unreachable (HTTP \(code))."))
                }
                return
            }
            consecutiveStatusFailures = 0
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
            if !silent {
                showToast(L("已连接 \(gpuName)", "Connected to \(gpuName)"))
                statusMessage = isSelectedModelLoadedInVRAM
                    ? L("已连接 \(gpuName) · \(activeModelSpec.name) 已驻留显存！", "Connected to \(gpuName) · \(activeModelSpec.name) ready in VRAM!")
                    : L("已连接 \(gpuName) · 可下载并加载模型至显存。", "Connected to \(gpuName) · Ready to download & load models.")
            }
        } catch {
            consecutiveStatusFailures += 1
            if !silent {
                resetStaleRemoteRuntimeState(reason: L("连接失败: \(error.localizedDescription)", "Connection failed: \(error.localizedDescription)"))
            } else if consecutiveStatusFailures >= 6 && isConnected {
                resetStaleRemoteRuntimeState(reason: L("与云端连接中断，可点「连接」重试。", "Lost connection to cloud server."))
            }
        }
    }

    func startBackgroundTelemetryLoop() {
        heartbeatTask?.cancel()
        heartbeatTask = Task {
            if !normalizedBaseURL.isEmpty {
                await checkColabStatus(silent: true)
            }
            var tick = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                tick += 1
                if isConnected {
                    if !isGenerating || tick % 5 == 0 {
                        await checkColabStatus(silent: true)
                    }
                } else if !normalizedBaseURL.isEmpty && tick % 4 == 0 {
                    await checkColabStatus(silent: true)
                }

                if tick % 16 == 0 && cliLoggedIn && !isWakingColab {
                    let shouldKeepalive = isConnected && (
                        isGenerating || isDownloadingWeights || isLoadingModel || comfyRunning || comfyStarting || comfyQueueRunning > 0 || idleTimeoutMinutes <= 0 || idleRemainingSeconds > 75
                    )
                    if shouldKeepalive {
                        let st = await ColabNativeBridge.shared.keepalive()
                        if let bal = st.paidBalance { officialCUBalance = bal }
                        if let rate = st.burnRateHourly { officialBurnRate = rate }
                        activeVMCount = st.activeAssignments
                    } else {
                        await refreshOfficialColabAccountUsage()
                    }
                }
            }
        }
    }

    func startRemoteDownloadWeights() async {
        let base = normalizedBaseURL
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/download_model") else { return }
        var targetModel = selectedModelId
        if activeModelSpec.isCustom, let regURL = URL(string: "\(base)/minimax_api/custom_model/register") {
            let curSpec = activeModelSpec
            var regReq = URLRequest(url: regURL)
            regReq.httpMethod = "POST"
            regReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
            regReq.httpBody = try? JSONSerialization.data(withJSONObject: [
                "repo_id": curSpec.repoId,
                "include_pattern": curSpec.includePattern,
                "custom_name": curSpec.nameZh,
                "modality_hint": "auto",
                "hf_token": hfToken,
                "auto_download": false
            ])
            if let (rData, _) = try? await URLSession.shared.data(for: regReq),
               let rDec = try? JSONDecoder().decode(ColabStatusResponse.self, from: rData),
               let regDto = rDec.registered_model {
                applyStatusResponse(rDec)
                targetModel = regDto.id
                selectedModelId = regDto.id
            }
        }
        isDownloadingWeights = true
        modelTaskProgress = 0.05
        statusMessage = L("正在启动云端 8 线程高速下载 \(activeModelSpec.name)...", "Starting 8-thread cloud download for \(activeModelSpec.name)...")
        do {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "model_id": targetModel,
                "hf_token": hfToken,
                "repo_id": activeModelSpec.isCustom ? activeModelSpec.repoId : "",
                "include_pattern": activeModelSpec.isCustom ? activeModelSpec.includePattern : "",
                "custom_name": activeModelSpec.isCustom ? activeModelSpec.nameZh : ""
            ])
            let (data, _) = try await URLSession.shared.data(for: req)
            applyStatusResponse(try JSONDecoder().decode(ColabStatusResponse.self, from: data))
            while isDownloadingWeights {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                if let sURL = URL(string: "\(base)/minimax_api/status?model_id=\(targetModel)"),
                   let (sData, _) = try? await URLSession.shared.data(from: sURL),
                   let sDec = try? JSONDecoder().decode(ColabStatusResponse.self, from: sData) {
                    applyStatusResponse(sDec)
                    if let msg = sDec.task_message, !msg.isEmpty { statusMessage = localizeServerText(msg) }
                    if !(sDec.is_downloading ?? false) { isDownloadingWeights = false; break }
                }
            }
            if allDownloaded {
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                showToast(L("权重下载完成！请点「加载到显存」", "Weights Ready! Tap 'Load to VRAM'"))
            }
        } catch {
            isDownloadingWeights = false
            statusMessage = L("下载权重请求失败: \(error.localizedDescription)", "Weight download failed: \(error.localizedDescription)")
        }
    }

    func loadRemoteModelToVRAM() async {
        let base = normalizedBaseURL
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/load_model") else { return }
        guard allDownloaded else {
            statusMessage = L("请先点击「下载权重」完成模型拉取。", "Please tap 'Download Weights' first.")
            return
        }
        let targetModel = selectedModelId
        isLoadingModel = true
        modelTaskProgress = 0.15
        statusMessage = L("正在装载 \(activeModelSpec.name) 至云端 GPU 显存...", "Loading \(activeModelSpec.name) into cloud GPU VRAM...")
        do {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "model_id": targetModel,
                "precision_mode": "bf16",
                "offload_mode": selectedOffload,
                "context_length": contextLengthTokens,
                "hf_token": hfToken,
                "repo_id": activeModelSpec.isCustom ? activeModelSpec.repoId : "",
                "include_pattern": activeModelSpec.isCustom ? activeModelSpec.includePattern : "",
                "custom_name": activeModelSpec.isCustom ? activeModelSpec.nameZh : ""
            ])
            let (data, _) = try await URLSession.shared.data(for: req)
            applyStatusResponse(try JSONDecoder().decode(ColabStatusResponse.self, from: data))
            while isLoadingModel {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                if let sURL = URL(string: "\(base)/minimax_api/status?model_id=\(targetModel)"),
                   let (sData, _) = try? await URLSession.shared.data(from: sURL),
                   let sDec = try? JSONDecoder().decode(ColabStatusResponse.self, from: sData) {
                    applyStatusResponse(sDec)
                    if let msg = sDec.task_message, !msg.isEmpty { statusMessage = localizeServerText(msg) }
                    if !(sDec.is_loading_vram ?? false) { isLoadingModel = false; break }
                }
            }
            if isSelectedModelLoadedInVRAM {
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                showToast(L("\(activeModelSpec.shortTitle) 已加载至显存！", "\(activeModelSpec.shortTitle) Loaded in VRAM!"))
            }
        } catch {
            isLoadingModel = false
            statusMessage = L("加载模型失败: \(error.localizedDescription)", "Load model failed: \(error.localizedDescription)")
        }
    }

    func unloadRemoteModelFromVRAM(modelId: String) async {
        let base = normalizedBaseURL
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/unload_model") else { return }
        isUnloadingModel = true
        defer { isUnloadingModel = false }
        do {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.timeoutInterval = 20
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["model_id": modelId])
            let (data, resp) = try await URLSession.shared.data(for: req)
            if (resp as? HTTPURLResponse)?.statusCode == 200,
               let decoded = try? JSONDecoder().decode(ColabStatusResponse.self, from: data) {
                applyStatusResponse(decoded)
                showToast(L("已释放显存 · 可用 \(String(format: "%.1f", vramFreeGB)) GB", "VRAM Freed · \(String(format: "%.1f", vramFreeGB)) GB Free"))
            }
        } catch {}
    }

    func updateIdleTimeoutConfig(_ minutes: Int) async {
        idleTimeoutMinutes = minutes
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/config_idle") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["idle_timeout_minutes": minutes])
        if let (data, _) = try? await URLSession.shared.data(for: req),
           let dec = try? JSONDecoder().decode(ColabStatusResponse.self, from: data) {
            applyStatusResponse(dec)
        }
    }

    func syncContextLengthToColab() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/config_context") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["context_length": contextLengthTokens])
        if let (data, _) = try? await URLSession.shared.data(for: req),
           let dec = try? JSONDecoder().decode(ColabStatusResponse.self, from: data) {
            applyStatusResponse(dec)
        }
    }

    func regenerateApiKey() {
        apiAccessKey = "sk-colab-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(20).lowercased()
        Task { await syncApiKeyToColab() }
    }

    func syncApiKeyToColab() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/config_api_key") else { return }
        let key = apiKeyEnabled ? apiAccessKey.trimmingCharacters(in: .whitespacesAndNewlines) : ""
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["api_key": key])
        if let (data, _) = try? await URLSession.shared.data(for: req),
           let dec = try? JSONDecoder().decode(ColabStatusResponse.self, from: data) {
            remoteApiKeySynced = true
            applyStatusResponse(dec)
        }
    }

    // MARK: - ComfyUI Cloud Actions

    func openComfyCanvas() {
        guard let url = URL(string: comfyPublicURL), !comfyPublicURL.isEmpty else { return }
        activeSafariURL = url
    }

    func startRemoteComfyUI(forceRestart: Bool = false) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/start") else {
            statusMessage = L("请先唤醒或连接云端实例后再启动 ComfyUI。", "Please connect to a cloud GPU runtime first.")
            return
        }
        comfyStarting = true
        comfyProgress = 0.05
        statusMessage = L("正在部署云端 ComfyUI 并挂载 Google Drive 持久化...", "Deploying cloud ComfyUI & mounting Google Drive...")
        do {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.timeoutInterval = 30
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "unload_existing": comfyUnloadOthersOnStart,
                "force_restart": forceRestart
            ])
            let (data, _) = try await URLSession.shared.data(for: req)
            applyStatusResponse(try JSONDecoder().decode(ColabStatusResponse.self, from: data))
            for _ in 0..<180 {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await checkColabStatus(silent: true)
                statusMessage = localizeServerText(comfyStatusMessage)
                if comfyRunning && !comfyPublicURL.isEmpty && !comfyStarting {
                    UINotificationFeedbackGenerator().notificationOccurred(.success)
                    showToast(L("ComfyUI 画布已就绪！", "ComfyUI Canvas Ready!"))
                    return
                }
                if !comfyStarting && !comfyRunning { break }
            }
        } catch {
            comfyStarting = false
            statusMessage = L("启动 ComfyUI 失败: \(error.localizedDescription)", "Failed to start ComfyUI: \(error.localizedDescription)")
        }
    }

    func stopRemoteComfyUI() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/stop") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 35
        if let (data, _) = try? await URLSession.shared.data(for: req),
           let dec = try? JSONDecoder().decode(ColabStatusResponse.self, from: data) {
            applyStatusResponse(dec)
        }
    }

    func syncRemoteComfyUIToDrive() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/sync_drive") else { return }
        comfySyncingDrive = true
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 90
        if let (data, _) = try? await URLSession.shared.data(for: req),
           let dec = try? JSONDecoder().decode(ColabStatusResponse.self, from: data) {
            applyStatusResponse(dec)
            showToast(L("已同步至 Google Drive", "Synced to Google Drive"))
        } else {
            comfySyncingDrive = false
        }
    }

    func installComfyPresetPack(_ pack: ComfyPresetPackItem) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/install_pack") else { return }
        comfyDownloadingWeight = true
        comfyInstallingPackId = pack.id
        comfyProgress = 0.04
        statusMessage = L("正在一键配齐「\(pack.localizedTitle)」...", "Installing preset pack \"\(pack.localizedTitle)\"...")
        do {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.timeoutInterval = 30
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "pack_id": pack.id,
                "auto_start": true,
                "hf_token": hfToken
            ])
            let (data, _) = try await URLSession.shared.data(for: req)
            applyStatusResponse(try JSONDecoder().decode(ColabStatusResponse.self, from: data))
            for _ in 0..<600 {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await checkColabStatus(silent: true)
                statusMessage = localizeServerText(comfyStatusMessage)
                if !comfyDownloadingWeight && comfyInstallingPackId.isEmpty && !comfyStarting { break }
            }
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        } catch {
            comfyDownloadingWeight = false
            comfyInstallingPackId = ""
            statusMessage = error.localizedDescription
        }
    }

    func startComfyWeightDownload() async {
        let src = comfyWeightSourceInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let base = normalizedBaseURL
        guard !src.isEmpty, isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/download_weight") else { return }
        comfyDownloadingWeight = true
        do {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "url_or_repo": src,
                "filename": comfyWeightFilenameInput.trimmingCharacters(in: .whitespacesAndNewlines),
                "subfolder": comfyWeightSubfolder,
                "hf_token": hfToken
            ])
            let (data, _) = try await URLSession.shared.data(for: req)
            applyStatusResponse(try JSONDecoder().decode(ColabStatusResponse.self, from: data))
            while comfyDownloadingWeight {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await checkColabStatus(silent: true)
            }
        } catch {
            comfyDownloadingWeight = false
        }
    }

    func pullComfyOutputToPhone(_ item: ComfyOutputMediaItem) async {
        let base = normalizedBaseURL
        let encodedRel = item.rel_path
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/output_media/\(encodedRel)") else { return }
        comfyPullingOutputRelPath = item.rel_path
        defer { comfyPullingOutputRelPath = nil }
        await downloadMediaToLocal(from: url, filename: "comfy_" + (item.filename as NSString).lastPathComponent, elapsedSeconds: nil)
    }


    // MARK: - Local Gallery & Photos Album Integration

    func refreshLocalMedia() {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: localSaveDir,
            includingPropertiesForKeys: [.creationDateKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        let validExts: Set<String> = ["mp4", "png", "jpg", "jpeg", "webp"]
        let items: [LocalMediaItem] = urls.compactMap { url in
            let ext = url.pathExtension.lowercased()
            guard validExts.contains(ext) else { return nil }
            let vals = try? url.resourceValues(forKeys: [.creationDateKey, .fileSizeKey])
            let date = vals?.creationDate ?? Date.distantPast
            let bytes = Double(vals?.fileSize ?? 0)
            let fname = url.lastPathComponent
            return LocalMediaItem(
                url: url,
                filename: fname,
                isImage: ext != "mp4",
                createdAt: date,
                elapsedSeconds: mediaElapsedMap[fname],
                fileSizeMB: bytes / 1_048_576.0
            )
        }.sorted(by: { $0.createdAt > $1.createdAt })

        localMediaItems = items
        if let cur = currentMediaURL, !FileManager.default.fileExists(atPath: cur.path) {
            player?.pause()
            player = nil
            currentPreviewImage = nil
            currentMediaURL = nil
        }
        if currentMediaURL == nil, let first = items.first {
            selectMedia(first.url, autoPlay: false)
        }
    }

    func selectMedia(_ url: URL, autoPlay: Bool = true) {
        currentMediaURL = url
        if url.pathExtension.lowercased() == "mp4" {
            currentPreviewImage = nil
            player = AVPlayer(url: url)
            if autoPlay { player?.play() }
        } else {
            player?.pause()
            player = nil
            currentPreviewImage = UIImage(contentsOfFile: url.path)
        }
    }

    func deleteLocalMedia(_ item: LocalMediaItem) {
        if currentMediaURL?.standardizedFileURL == item.url.standardizedFileURL || currentMediaURL?.lastPathComponent == item.filename {
            player?.pause()
            player = nil
            currentPreviewImage = nil
            currentMediaURL = nil
        }
        mediaElapsedMap.removeValue(forKey: item.filename)
        do {
            if FileManager.default.fileExists(atPath: item.url.path) {
                try FileManager.default.removeItem(at: item.url)
            }
            refreshLocalMedia()
            UINotificationFeedbackGenerator().notificationOccurred(.success)
            showToast(L("已从作品库删除: \(item.filename)", "Deleted: \(item.filename)"))
        } catch {
            refreshLocalMedia()
            UINotificationFeedbackGenerator().notificationOccurred(.error)
            showToast(L("删除失败: \(error.localizedDescription)", "Delete failed: \(error.localizedDescription)"))
        }
    }

    func shareLocalMedia(_ url: URL) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            UINotificationFeedbackGenerator().notificationOccurred(.error)
            showToast(L("文件不存在或已被删除", "File no longer exists"))
            refreshLocalMedia()
            return
        }
        presentSystemShareSheet(items: [url])
    }

    func downloadMediaToLocal(from remoteURL: URL, filename: String, elapsedSeconds: Double?) async {
        let alreadyExists = FileManager.default.fileExists(atPath: localSaveDir.appendingPathComponent(filename).path)
        do {
            let (tempURL, resp) = try await URLSession.shared.download(from: remoteURL)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else { return }
            let destURL = localSaveDir.appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: destURL.path) {
                try? FileManager.default.removeItem(at: destURL)
            }
            try FileManager.default.moveItem(at: tempURL, to: destURL)
            if let el = elapsedSeconds, el > 0 {
                mediaElapsedMap[filename] = el
            }
            refreshLocalMedia()
            selectMedia(destURL, autoPlay: true)

            if autoSaveToPhotos && !alreadyExists {
                await saveMediaToSystemPhotos(url: destURL, silent: true)
            } else {
                showToast(L("已下载成片: \(filename)", "Saved: \(filename)"))
            }
        } catch {
            statusMessage = L("下载成片失败: \(error.localizedDescription)", "Failed to download media: \(error.localizedDescription)")
        }
    }

    func saveMediaToSystemPhotos(url: URL, silent: Bool = false) async {
        guard FileManager.default.fileExists(atPath: url.path) else {
            if !silent {
                UINotificationFeedbackGenerator().notificationOccurred(.error)
                showToast(L("文件不存在或已被删除", "File no longer exists"))
                refreshLocalMedia()
            }
            return
        }

        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            if !silent {
                UINotificationFeedbackGenerator().notificationOccurred(.warning)
                showToast(L("请在系统「设置 -> EasyColab -> 照片」中允许访问相册", "Allow Photos access in iOS Settings -> EasyColab"))
            }
            return
        }

        let ext = url.pathExtension.lowercased()
        let isVideo = ext == "mp4"
        var tempVideoURL: URL? = nil
        defer {
            if let tempVideoURL {
                try? FileManager.default.removeItem(at: tempVideoURL)
            }
        }

        do {
            if isVideo {
                let stagedURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("easycolab_export_\(UUID().uuidString).mp4")
                try FileManager.default.copyItem(at: url, to: stagedURL)
                tempVideoURL = stagedURL

                try await PHPhotoLibrary.shared().performChanges {
                    let req = PHAssetCreationRequest.forAsset()
                    let opts = PHAssetResourceCreationOptions()
                    opts.originalFilename = url.lastPathComponent
                    req.addResource(with: .video, fileURL: stagedURL, options: opts)
                }
            } else {
                let imageData: Data
                let exportName: String
                if (ext == "png" || ext == "jpg" || ext == "jpeg"),
                   let raw = try? Data(contentsOf: url), !raw.isEmpty {
                    imageData = raw
                    exportName = url.lastPathComponent
                } else if let uiImg = UIImage(contentsOfFile: url.path),
                          let png = uiImg.pngData() {
                    imageData = png
                    exportName = url.deletingPathExtension().lastPathComponent + ".png"
                } else {
                    throw NSError(
                        domain: "EasyColabPhotos",
                        code: -2,
                        userInfo: [NSLocalizedDescriptionKey: L("无法读取图像数据", "Unable to decode image file")]
                    )
                }

                try await PHPhotoLibrary.shared().performChanges {
                    let req = PHAssetCreationRequest.forAsset()
                    let opts = PHAssetResourceCreationOptions()
                    opts.originalFilename = exportName
                    req.addResource(with: .photo, data: imageData, options: opts)
                }
            }

            if !silent {
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            }
            showToast(isVideo ? L("视频已存入系统相册！", "Video saved to Photos!") : L("高清图已存入系统相册！", "Image saved to Photos!"))
        } catch {
            if !silent {
                UINotificationFeedbackGenerator().notificationOccurred(.error)
                showToast(L("保存到相册失败: \(error.localizedDescription)", "Save to Photos failed: \(error.localizedDescription)"))
            }
        }
    }
}
