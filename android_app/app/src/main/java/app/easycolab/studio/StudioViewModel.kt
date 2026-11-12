package app.easycolab.studio

import android.app.Activity
import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import android.widget.Toast
import androidx.browser.customtabs.CustomTabsIntent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.URI
import java.net.URLEncoder
import java.util.Locale
import java.util.UUID
import java.util.concurrent.TimeUnit

enum class AppTab(val index: Int) {
    CREATE(0),
    CLOUD(1),
    GALLERY(2),
    SETTINGS(3)
}

class StudioViewModel(application: Application) : AndroidViewModel(application) {

    internal val appContext: Context = application.applicationContext
    internal val prefs = appContext.getSharedPreferences("easycolab_prefs", Context.MODE_PRIVATE)
    internal val bridge = ColabNativeBridge.getInstance(appContext)

    internal val apiClient = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    val localGalleryDir: File = File(appContext.filesDir, "gallery").apply { mkdirs() }

    var selectedTab by mutableStateOf(AppTab.CREATE)

    // Preferences & Privacy
    var maskServerURL by mutableStateOf(prefs.getBoolean("maskServerURL", false))
        private set

    fun updateMaskServerURL(enabled: Boolean) {
        maskServerURL = enabled
        prefs.edit().putBoolean("maskServerURL", enabled).apply()
    }

    var colabServerURL by mutableStateOf(prefs.getString("colabServerURL", "") ?: "")
        private set

    fun updateColabServerURL(url: String) {
        colabServerURL = url
        prefs.edit().putString("colabServerURL", url).apply()
    }

    var hfToken by mutableStateOf(prefs.getString("hfToken", "") ?: "")
        private set

    fun updateHfToken(token: String) {
        hfToken = token
        prefs.edit().putString("hfToken", token).apply()
    }

    var autoSaveToGallery by mutableStateOf(prefs.getBoolean("autoSaveToGallery", true))
        private set

    fun updateAutoSaveToGallery(enabled: Boolean) {
        autoSaveToGallery = enabled
        prefs.edit().putBoolean("autoSaveToGallery", enabled).apply()
    }

    var showOnboardingDialog by mutableStateOf(!prefs.getBoolean("hasSeenAndroidOnboardingV1", false))
        private set

    fun setOnboardingVisible(visible: Boolean) {
        showOnboardingDialog = visible
        if (!visible) {
            prefs.edit().putBoolean("hasSeenAndroidOnboardingV1", true).apply()
        }
    }

    // Native Google OAuth & Cloud Billing State
    var cliLoggedIn by mutableStateOf(false)
    var isAuthenticatingGoogle by mutableStateOf(false)
    var accountEmail by mutableStateOf("")
    var showOAuthCodeDialog by mutableStateOf(false)
    var oauthCodeText by mutableStateOf("")
    var officialCUBalance by mutableStateOf<Double?>(null)
    var officialBurnRate by mutableStateOf<Double?>(null)
    var activeVMCount by mutableIntStateOf(0)
    var hasCustomOAuth by mutableStateOf(bridge.hasCustomOAuthConfig())

    var wakeGPUChoice by mutableStateOf(
        (prefs.getString("wakeGPUChoiceV2", "G4") ?: "G4").let {
            if (it in listOf("G4", "H100", "A100", "L4", "T4")) it else "G4"
        }
    )
        private set

    fun updateWakeGPUChoice(gpu: String) {
        wakeGPUChoice = gpu
        prefs.edit().putString("wakeGPUChoiceV2", gpu).apply()
    }

    var isWakingColab by mutableStateOf(false)
    var showDisconnectConfirm by mutableStateOf(false)

    // Remote GPU & Telemetry State
    var isConnected by mutableStateOf(false)
    var gpuName by mutableStateOf(L("未连接 Colab 服务端", "Colab Server Disconnected"))
    var vramFreeGB by mutableDoubleStateOf(0.0)
    var vramTotalGB by mutableDoubleStateOf(0.0)
    var vramUsedGB by mutableDoubleStateOf(0.0)
    var ramTotalGB by mutableDoubleStateOf(0.0)
    var ramUsedGB by mutableDoubleStateOf(0.0)
    var ramFreeGB by mutableDoubleStateOf(0.0)
    var diskTotalGB by mutableDoubleStateOf(0.0)
    var diskUsedGB by mutableDoubleStateOf(0.0)
    var diskFreeGB by mutableDoubleStateOf(0.0)
    var driveTotalGB by mutableDoubleStateOf(0.0)
    var driveUsedGB by mutableDoubleStateOf(0.0)
    var driveFreeGB by mutableDoubleStateOf(0.0)
    val vramHistoryPct = mutableStateListOf<Float>()
    val ramHistoryPct = mutableStateListOf<Float>()
    val diskHistoryPct = mutableStateListOf<Float>()
    var combinedTelemetryChartMode by mutableStateOf(true)

    // Active Model & VRAM Residency
    var selectedModelId by mutableStateOf("minimax_h3")
    var customModels by mutableStateOf<List<OpenModelSpec>>(emptyList())
    var showAddCustomModelDialog by mutableStateOf(false)
    var customRepoInput by mutableStateOf("")
    var customIncludePatternInput by mutableStateOf("")
    var customDisplayNameInput by mutableStateOf("")
    var customModalityInput by mutableStateOf("auto")
    var isRegisteringCustomModel by mutableStateOf(false)
    var customModelErrorMessage by mutableStateOf("")
    var trendingHFModels by mutableStateOf<List<HFTrendingModelItem>>(emptyList())
    var trendingHFCategory by mutableStateOf("all")
    var isLoadingTrendingHF by mutableStateOf(false)
    var activeHFModelDetail by mutableStateOf<HFModelDetailInfo?>(null)
    var isLoadingHFModelDetail by mutableStateOf(false)
    var showHFModelDetailDialog by mutableStateOf(false)
    var loadedModelId by mutableStateOf<String?>(null)
    var loadedModelIds by mutableStateOf<List<String>>(emptyList())
    var loadedModelsDetail by mutableStateOf<List<LoadedModelSummary>>(emptyList())
    var modelLoaded by mutableStateOf(false)
    var isUnloadingModel by mutableStateOf(false)
    var currentModelName by mutableStateOf(L("未加载", "Not Loaded"))

    var uptimeSeconds by mutableIntStateOf(0)
    var cuBurnRatePerHour by mutableDoubleStateOf(0.0)
    var sessionCUUsed by mutableDoubleStateOf(0.0)
    var idleTimeoutMinutes by mutableIntStateOf(
        prefs.getInt("idleTimeoutMinutes", 30).let { if (it in listOf(0, 15, 30, 60)) it else 30 }
    )
        private set
    var idleRemainingSeconds by mutableIntStateOf(1800)
    var drivePersisted by mutableStateOf(false)
    val liveConsoleLines = mutableStateListOf(
        L("欢迎使用 EasyColab Android 多模态云端工作台", "Welcome to EasyColab Android Multimodal Cloud Studio"),
        L("纯 Kotlin 原生直连 Google Colab G4 (96GB)，支持一键免网页唤醒、生视频/生图自动存系统相册与闲置省额度保护。", "Pure-Kotlin native Colab G4 (96GB) client — 1-tap headless wake, auto-save to Gallery & idle CU protection.")
    )

    // Model Weights State
    var modelFiles by mutableStateOf(OPEN_MODEL_CATALOG[0].defaultFiles)
    var downloadedCount by mutableIntStateOf(0)
    var totalFileCount by mutableIntStateOf(OPEN_MODEL_CATALOG[0].defaultFiles.size)
    var allDownloaded by mutableStateOf(false)
    var totalDiskSizeGB by mutableDoubleStateOf(0.0)
    var isDownloadingWeights by mutableStateOf(false)
    var isLoadingModel by mutableStateOf(false)
    var modelTaskProgress by mutableFloatStateOf(0f)
    var selectedOffload by mutableStateOf("disk")

    // ComfyUI Cloud State
    var comfyRunning by mutableStateOf(false)
    var comfyStarting by mutableStateOf(false)
    var comfySyncingDrive by mutableStateOf(false)
    var comfyDownloadingWeight by mutableStateOf(false)
    var comfyInstallingPackId by mutableStateOf("")
    var comfyPresetPacks by mutableStateOf(DEFAULT_COMFY_PRESET_PACKS)
    var comfyProgress by mutableFloatStateOf(0f)
    var comfyStatusMessage by mutableStateOf(L("尚未启动云端 ComfyUI 工作台", "Cloud ComfyUI Studio Not Started"))
    var comfyPublicURL by mutableStateOf("")
    var comfyQueueRunning by mutableIntStateOf(0)
    var comfyQueuePending by mutableIntStateOf(0)
    var comfyPersistenceLabel by mutableStateOf(L("Google Drive 自动软链持久化", "Google Drive Auto-Symlink Persistence"))
    var comfyCustomNodesCount by mutableIntStateOf(0)
    var comfyModelsCount by mutableIntStateOf(0)
    var comfyWorkflowsCount by mutableIntStateOf(0)
    var comfyRecentOutputs by mutableStateOf<List<ComfyOutputMediaItem>>(emptyList())
    var comfyUnloadOthersOnStart by mutableStateOf(false)
    var comfyWeightSourceInput by mutableStateOf("")
    var comfyWeightFilenameInput by mutableStateOf("")
    var comfyWeightSubfolder by mutableStateOf("diffusion_models")
    var comfyPullingOutputRelPath by mutableStateOf<String?>(null)

    // Generation Parameters
    var taskModeId by mutableStateOf("t2va")
    var visualPrompt by mutableStateOf(PRESET_TEMPLATES[0].visual)
    var negativePrompt by mutableStateOf("")
    var soundscapePrompt by mutableStateOf(PRESET_TEMPLATES[0].soundscape)
    var musicPrompt by mutableStateOf(PRESET_TEMPLATES[0].music)
    var firstFrameBitmap by mutableStateOf<Bitmap?>(null)
    var lastFrameBitmap by mutableStateOf<Bitmap?>(null)
    var selectedVideoResolution by mutableStateOf(VIDEO_RESOLUTIONS[0])
    var selectedImageResolution by mutableStateOf(IMAGE_RESOLUTIONS[0])
    var numFrames by mutableFloatStateOf(124f)
    var inferenceSteps by mutableFloatStateOf(35f)
    var fps by mutableFloatStateOf(24f)
    var seed by mutableIntStateOf(42)
    var randomizeSeed by mutableStateOf(true)
    var batchCount by mutableIntStateOf(1)
    var denoisingStrength by mutableFloatStateOf(0.70f)

    // Qwen3.8 LLM/VLM & API Gateway
    var contextLengthK by mutableIntStateOf(prefs.getInt("qwenContextLengthK", 32).coerceIn(4, 256))
        private set

    fun updateContextLengthK(newVal: Int) {
        val clamped = newVal.coerceIn(4, 256)
        contextLengthK = clamped
        prefs.edit().putInt("qwenContextLengthK", clamped).apply()
    }

    var loadedContextLengthTokens by mutableIntStateOf(0)
    var enableThinking by mutableStateOf(true)
        private set

    fun updateEnableThinking(enabled: Boolean) {
        enableThinking = enabled
        if (enabled) {
            temperature = 1.0f
            topP = 0.95f
            presencePenalty = 0.0f
        } else {
            temperature = 0.7f
            topP = 0.80f
            presencePenalty = 1.5f
        }
    }

    var reasoningEffort by mutableStateOf("medium")
    var maxNewTokens by mutableFloatStateOf(8192f)
    var temperature by mutableFloatStateOf(1.0f)
    var topP by mutableFloatStateOf(0.95f)
    var presencePenalty by mutableFloatStateOf(0.0f)
    val chatHistory = mutableStateListOf<ChatTurnItem>()

    var apiKeyEnabled by mutableStateOf(prefs.getBoolean("apiKeyEnabled", true))
        private set

    fun updateApiKeyEnabled(enabled: Boolean) {
        apiKeyEnabled = enabled
        prefs.edit().putBoolean("apiKeyEnabled", enabled).apply()
        viewModelScope.launch { syncApiKeyToColab() }
    }

    var apiAccessKey by mutableStateOf(
        prefs.getString("apiAccessKey", null)?.takeIf { it.isNotBlank() } ?: run {
            val gen = "sk-colab-" + UUID.randomUUID().toString().replace("-", "").take(20).lowercase(Locale.US)
            prefs.edit().putString("apiAccessKey", gen).apply()
            gen
        }
    )
        private set

    fun updateApiAccessKey(newKey: String) {
        apiAccessKey = newKey
        prefs.edit().putString("apiAccessKey", newKey).apply()
    }

    var apiRequestCount by mutableIntStateOf(0)

    // Active Job & Media Preview
    var isGenerating by mutableStateOf(false)
    var progressValue by mutableFloatStateOf(0f)
    var currentJobElapsed by mutableDoubleStateOf(0.0)
    var lastCompletedElapsed by mutableStateOf<Double?>(null)
    var statusMessage by mutableStateOf(
        L(
            "就绪：可在「云算力」页绑定账号一键唤醒 G4，或粘贴已有 gradio.live 地址连接。",
            "Ready: Bind Google account in Cloud tab for 1-tap G4 wake, or paste an active gradio.live URL."
        )
    )
    var toastBanner by mutableStateOf<String?>(null)

    var currentMediaFile by mutableStateOf<File?>(null)
    var currentPreviewBitmap by mutableStateOf<Bitmap?>(null)
    var localMediaItems by mutableStateOf<List<LocalMediaItem>>(emptyList())

    internal var pendingJobId: String?
        get() = prefs.getString("pendingAndroidJobId", null)
        set(value) {
            if (value.isNullOrEmpty()) prefs.edit().remove("pendingAndroidJobId").apply()
            else prefs.edit().putString("pendingAndroidJobId", value).apply()
        }

    internal var pendingAssistantIndex: Int? = null
    internal var activeGenerationToken: String = UUID.randomUUID().toString()
    internal var heartbeatJob: Job? = null
    internal var generationTimerJob: Job? = null
    internal var toastJob: Job? = null
    internal var generationStartedAtMs: Long = 0L
    internal var consecutiveStatusFailures: Int = 0
    internal var lastTelemetrySampleMs: Long = 0L
    internal var remoteContextSynced = false
    internal var remoteIdleSynced = false
    internal var remoteApiKeySynced = false

    val normalizedBaseURL: String
        get() = colabServerURL.trim().trimEnd('/')

    val allAvailableModels: List<OpenModelSpec>
        get() = OPEN_MODEL_CATALOG + customModels

    val activeModelSpec: OpenModelSpec
        get() = allAvailableModels.firstOrNull { it.id == selectedModelId } ?: OPEN_MODEL_CATALOG[0]

    val activeModeItem: TaskModeItem
        get() = activeModelSpec.modes.firstOrNull { it.id == taskModeId } ?: activeModelSpec.modes[0]

    val isSelectedModelLoadedInVRAM: Boolean
        get() = loadedModelIds.contains(selectedModelId) || (modelLoaded && loadedModelId == selectedModelId)

    val contextLengthTokens: Int
        get() = contextLengthK.coerceIn(4, 256) * 1024

    val displayAccountLabel: String
        get() {
            if (!cliLoggedIn) return L("未绑定", "Unbound")
            val raw = accountEmail.trim()
            if (raw.isEmpty()) return L("已授权", "Authorized")
            if (maskServerURL && raw.contains("@")) {
                val prefix = raw.substringBefore("@").take(2)
                val domain = raw.substringAfter("@")
                return "$prefix•••@$domain"
            }
            return raw
        }

    val displayRemainingCU: Double
        get() = officialCUBalance ?: 0.0

    val displayBurnRate: Double
        get() {
            val rate = officialBurnRate
            if (rate != null && (rate > 0.0 || !isConnected)) return rate
            return cuBurnRatePerHour
        }

    val formattedUptime: String
        get() {
            val h = uptimeSeconds / 3600
            val m = (uptimeSeconds % 3600) / 60
            val s = uptimeSeconds % 60
            return String.format(Locale.US, "%02d:%02d:%02d", h, m, s)
        }

    val formattedIdleRemaining: String
        get() {
            if (idleTimeoutMinutes <= 0 || idleRemainingSeconds < 0) {
                return L("已关闭自动休眠", "Auto-sleep off")
            }
            val m = idleRemainingSeconds / 60
            val s = idleRemainingSeconds % 60
            return L(
                "${m}分${String.format(Locale.US, "%02d", s)}秒后释放",
                "Auto-stop in ${m}m ${String.format(Locale.US, "%02d", s)}s"
            )
        }

    fun maskedURLDisplay(rawURL: String): String {
        val trimmed = rawURL.trim()
        if (!maskServerURL || trimmed.isEmpty()) return trimmed
        return try {
            val uri = URI(trimmed)
            val scheme = uri.scheme ?: "https"
            val host = uri.host ?: return "https://••••••••"
            val parts = host.split(".")
            val suffix = if (parts.size >= 2) parts.takeLast(2).joinToString(".") else host
            "$scheme://••••••••.$suffix"
        } catch (_: Exception) {
            "https://••••••••"
        }
    }

    init {
        LanguageManager.init(appContext)
        loadPersistedCustomModels()
        refreshLocalMedia()
        startBackgroundTelemetryLoop()
        viewModelScope.launch {
            hasCustomOAuth = bridge.hasCustomOAuthConfig()
            refreshOfficialColabAccountUsage()
            if (!pendingJobId.isNullOrEmpty() && normalizedBaseURL.isNotEmpty()) {
                resumePendingJobIfNeeded()
            }
        }
    }

    fun showToast(message: String) {
        toastBanner = message
        toastJob?.cancel()
        toastJob = viewModelScope.launch {
            delay(2800)
            if (toastBanner == message) {
                toastBanner = null
            }
        }
    }

    internal fun appendConsoleLine(line: String) {
        val cleaned = stripEmoji(line)
        if (cleaned.isNotEmpty()) {
            liveConsoleLines.add(cleaned)
            if (liveConsoleLines.size > 160) {
                liveConsoleLines.removeRange(0, liveConsoleLines.size - 160)
            }
        }
    }

    internal fun loadElapsedMap(): MutableMap<String, Double> {
        val raw = prefs.getString("mediaElapsedJson", null) ?: return mutableMapOf()
        return try {
            val obj = JSONObject(raw)
            val map = mutableMapOf<String, Double>()
            val keys = obj.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                map[k] = obj.optDouble(k, 0.0)
            }
            map
        } catch (_: Exception) {
            mutableMapOf()
        }
    }

    internal fun recordMediaElapsed(filename: String, elapsed: Double) {
        val map = loadElapsedMap()
        map[filename] = elapsed
        val obj = JSONObject()
        for ((k, v) in map) obj.put(k, v)
        prefs.edit().putString("mediaElapsedJson", obj.toString()).apply()
    }

    // MARK: - Browser Helper (Chrome Custom Tabs -> System Browser Fallback)

    fun openExternalOrCustomTab(context: Context, urlString: String) {
        val clean = urlString.trim()
        if (clean.isEmpty()) return
        val uri = try { Uri.parse(clean) } catch (_: Exception) { return }
        try {
            val customTab = CustomTabsIntent.Builder()
                .setShowTitle(true)
                .build()
            customTab.launchUrl(context, uri)
        } catch (_: Exception) {
            try {
                val viewIntent = Intent(Intent.ACTION_VIEW, uri).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(viewIntent)
            } catch (e: Exception) {
                showToast(L("无法打开浏览器: ${e.message}", "Cannot open browser: ${e.message}"))
            }
        }
    }

    fun copyToClipboard(context: Context, label: String, text: String, toastText: String? = null) {
        val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
        cm?.setPrimaryClip(ClipData.newPlainText(label, text))
        toastText?.let { showToast(it) }
    }

    // MARK: - App Foreground Resume & Photo Picker Decoding

    fun onAppResumed(context: Context) {
        startBackgroundTelemetryLoop()
        if (showOAuthCodeDialog && oauthCodeText.isEmpty()) {
            val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
            val clip = cm?.primaryClip?.getItemAt(0)?.coerceToText(context)?.toString()?.trim().orEmpty()
            if (clip.startsWith("4/") && clip.length > 20 && !clip.contains(" ")) {
                oauthCodeText = clip
            }
        }
        viewModelScope.launch {
            if (normalizedBaseURL.isNotEmpty()) {
                checkColabStatus(silent = true)
            }
            if (cliLoggedIn) {
                refreshOfficialColabAccountUsage()
            }
            resumePendingJobIfNeeded()
        }
    }

    internal suspend fun resumePendingJobIfNeeded() {
        val jobId = pendingJobId?.takeIf { it.isNotEmpty() } ?: return
        val base = normalizedBaseURL.takeIf { it.isNotEmpty() } ?: return
        if (isGenerating) return
        val tok = UUID.randomUUID().toString()
        activeGenerationToken = tok
        isGenerating = true
        startLiveGenerationTimer()
        pollColabJob(base, jobId, pendingAssistantIndex, tok)
    }

    fun pickKeyframeFromUri(context: Context, uri: Uri?, isFirstFrame: Boolean) {
        if (uri == null) return
        viewModelScope.launch {
            val bmp = withContext(Dispatchers.IO) { decodeSampledBitmap(context, uri, maxEdge = 1920) }
            if (bmp != null) {
                if (isFirstFrame) firstFrameBitmap = bmp else lastFrameBitmap = bmp
            } else {
                showToast(L("读取所选图片失败", "Failed to decode selected photo"))
            }
        }
    }

    internal fun decodeSampledBitmap(context: Context, uri: Uri, maxEdge: Int): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val src = ImageDecoder.createSource(context.contentResolver, uri)
                ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    decoder.isMutableRequired = false
                    val w = info.size.width
                    val h = info.size.height
                    val longest = maxOf(w, h)
                    if (longest > maxEdge) {
                        val scale = maxEdge.toFloat() / longest.toFloat()
                        decoder.setTargetSize(
                            (w * scale).toInt().coerceAtLeast(1),
                            (h * scale).toInt().coerceAtLeast(1)
                        )
                    }
                }
            } else {
                context.contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
            }
        } catch (_: Exception) {
            null
        }
    }

    // MARK: - Native Google OAuth2 PKCE (Chrome Custom Tabs + Reverse Client Scheme)

    suspend fun refreshOfficialColabAccountUsage() {
        val res = bridge.fetchAuthStatus()
        cliLoggedIn = res.loggedIn
        if (res.email.isNotEmpty()) {
            accountEmail = res.email
        } else if (!res.loggedIn) {
            accountEmail = ""
        }
        res.paidBalance?.let { officialCUBalance = it }
        res.burnRateHourly?.let { officialBurnRate = it }
        activeVMCount = res.activeAssignments
        if (res.activeAssignments == 0 && !isConnected && (loadedModelsDetail.isNotEmpty() || vramTotalGB > 0.0)) {
            resetStaleRemoteRuntimeState(
                L("云端实例当前未运行，已清理旧显存记录。", "No active cloud runtime detected; cleared stale VRAM state.")
            )
        }
    }

    fun startGoogleOAuthLogin(context: Context) {
        if (isAuthenticatingGoogle) return
        viewModelScope.launch {
            showOAuthCodeDialog = false
            try {
                val sessionReq = bridge.generateAuthSession(preferCustomScheme = true)
                if (sessionReq.callbackScheme == null) {
                    // Custom Web OAuth JSON without Android reverse scheme -> open manual dialog + browser
                    oauthCodeText = ""
                    showOAuthCodeDialog = true
                    openExternalOrCustomTab(context, sessionReq.authUrl)
                } else {
                    statusMessage = L(
                        "请在弹出的 Google 页面登录并点击允许，授权后将自动返回 App...",
                        "Sign in on the Google page and tap Allow — you will return to the app automatically..."
                    )
                    openExternalOrCustomTab(context, sessionReq.authUrl)
                }
            } catch (e: Exception) {
                statusMessage = L("生成登录会话失败: ${e.message}", "Failed to start auth session: ${e.message}")
            }
        }
    }

    fun handleOAuthCallbackIntent(intent: Intent?): Boolean {
        val uri = intent?.data ?: return false
        val scheme = uri.scheme.orEmpty()
        if (!scheme.startsWith("com.googleusercontent.apps.")) return false
        val fullCallback = uri.toString()
        viewModelScope.launch {
            isAuthenticatingGoogle = true
            statusMessage = L("正在验证 Google 授权令牌...", "Verifying Google authorization token...")
            val res = bridge.exchangeAuthCode(fullCallback)
            isAuthenticatingGoogle = false
            if (res.ok && res.loggedIn) {
                showOAuthCodeDialog = false
                oauthCodeText = ""
                refreshOfficialColabAccountUsage()
                val welcome = if (accountEmail.isEmpty()) {
                    L("Google 账号绑定成功！", "Google Account Bound!")
                } else {
                    L("已绑定: $accountEmail", "Signed in: $accountEmail")
                }
                statusMessage = welcome
                showToast(welcome)
            } else {
                val err = res.error ?: "OAuth Error"
                statusMessage = L("授权未完成: $err", "Sign-in incomplete: $err")
                showToast(L("授权失败: $err", "Auth failed: $err"))
            }
        }
        return true
    }

    fun openManualOAuthDialog(context: Context?) {
        viewModelScope.launch {
            try {
                val req = bridge.generateAuthSession(preferCustomScheme = false)
                oauthCodeText = ""
                showOAuthCodeDialog = true
                if (context != null) {
                    openExternalOrCustomTab(context, req.authUrl)
                }
            } catch (e: Exception) {
                statusMessage = L("生成手动授权链接失败: ${e.message}", "Failed to create auth URL: ${e.message}")
            }
        }
    }

    fun copyManualOAuthLink(context: Context) {
        viewModelScope.launch {
            try {
                val req = bridge.generateAuthSession(preferCustomScheme = false)
                copyToClipboard(
                    context,
                    "Google OAuth URL",
                    req.authUrl,
                    L("已复制登录链接，可粘贴到任意浏览器打开", "Auth URL copied — open in any browser")
                )
            } catch (_: Exception) {
            }
        }
    }

    fun submitManualOAuthCode() {
        val code = oauthCodeText.trim()
        if (code.isEmpty()) return
        viewModelScope.launch {
            isAuthenticatingGoogle = true
            val res = bridge.exchangeAuthCode(code)
            isAuthenticatingGoogle = false
            if (res.ok && res.loggedIn) {
                showOAuthCodeDialog = false
                oauthCodeText = ""
                refreshOfficialColabAccountUsage()
                val welcome = if (accountEmail.isEmpty()) {
                    L("Google 账号绑定成功！", "Google Account Bound!")
                } else {
                    L("已绑定: $accountEmail", "Signed in: $accountEmail")
                }
                statusMessage = welcome
                showToast(welcome)
            } else {
                statusMessage = L("授权码验证失败: ${res.error ?: "Invalid code"}", "Code verification failed: ${res.error ?: "Invalid code"}")
            }
        }
    }

    fun logoutGoogleAccount() {
        viewModelScope.launch {
            bridge.logout()
            cliLoggedIn = false
            accountEmail = ""
            officialCUBalance = null
            officialBurnRate = null
            activeVMCount = 0
            statusMessage = L("已解绑本地 Google 授权凭证。", "Unbound local Google OAuth credentials.")
        }
    }

    fun saveCustomOAuthConfig(rawJson: String) {
        viewModelScope.launch {
            try {
                val enabled = bridge.saveCustomOAuthJson(rawJson)
                hasCustomOAuth = enabled
                showToast(
                    if (enabled) L("已保存自定义 OAuth 客户端配置", "Saved custom OAuth client JSON")
                    else L("已恢复内置默认 OAuth 配置", "Restored built-in OAuth client")
                )
            } catch (e: Exception) {
                showToast(e.message ?: "Invalid JSON")
            }
        }
    }

    // MARK: - Headless Wake & Cloud Shutdown

    fun wakeColabHeadless() {
        if (!cliLoggedIn) {
            statusMessage = L("请先点击「一键绑定 Google 账号」完成授权。", "Please sign in with Google first.")
            return
        }
        if (isWakingColab) return

        viewModelScope.launch {
            val appPyBytes = try {
                withContext(Dispatchers.IO) {
                    appContext.assets.open("app.py").use { it.readBytes() }
                }
            } catch (e: Exception) {
                statusMessage = L("读取内置 app.py 失败: ${e.message}", "Missing bundled app.py in APK assets: ${e.message}")
                return@launch
            }

            isWakingColab = true
            statusMessage = L(
                "正在云端免网页申请 $wakeGPUChoice GPU 并部署多模态服务端...",
                "Allocating $wakeGPUChoice GPU & deploying server headlessly..."
            )
            appendConsoleLine(L("=== 启动原生免网页唤醒 ($wakeGPUChoice) ===", "=== Starting Native Headless Wake ($wakeGPUChoice) ==="))

            try {
                val tunnelUrl = bridge.wakeAndDeploy(wakeGPUChoice, appPyBytes) { logLine ->
                    viewModelScope.launch {
                        val localized = localizeServerText(stripEmoji(logLine))
                        appendConsoleLine(localized)
                        statusMessage = localized
                    }
                }
                isWakingColab = false
                updateColabServerURL(tunnelUrl)
                showToast(L("云端 $wakeGPUChoice 已唤醒并自动连接！", "Cloud $wakeGPUChoice Ready & Connected!"))
                checkColabStatus(silent = false)
                refreshOfficialColabAccountUsage()
            } catch (e: Exception) {
                isWakingColab = false
                val msg = e.message ?: "Wake error"
                appendConsoleLine(L("唤醒失败: $msg", "Wake failed: $msg"))
                statusMessage = L("唤醒失败: $msg", "Wake failed: $msg")
                refreshOfficialColabAccountUsage()
            }
        }
    }

    fun disconnectRemoteColab() {
        viewModelScope.launch {
            statusMessage = L("正在释放全部云端 Colab 实例并归零计费...", "Terminating all cloud Colab VMs & stopping billing...")
            val base = normalizedBaseURL
            if (base.isNotEmpty()) {
                withContext(Dispatchers.IO) {
                    try {
                        val req = Request.Builder()
                            .url("$base/minimax_api/disconnect")
                            .post("{}".toRequestBody("application/json".toMediaType()))
                            .build()
                        apiClient.newCall(req).execute().close()
                    } catch (_: Exception) {
                    }
                }
            }
            if (cliLoggedIn) {
                val res = bridge.stopAll()
                res.paidBalance?.let { officialCUBalance = it }
                officialBurnRate = res.burnRateHourly ?: 0.0
                activeVMCount = res.activeAssignments
                statusMessage = res.message ?: L("已彻底释放全部云端实例，计费归零。", "All cloud instances terminated (0 CU/h).")
                showToast(L("已释放实例 · 0.00 CU/h", "Instances Terminated · 0 CU/h"))
            } else {
                statusMessage = L("已向服务端发送释放关机指令。", "Sent shutdown command to cloud runtime.")
            }
            pendingJobId = null
            resetStaleRemoteRuntimeState("")
        }
    }

    // MARK: - Telemetry & Model Management

    internal fun recordTelemetrySample() {
        val now = System.currentTimeMillis()
        if (now - lastTelemetrySampleMs < 850L) return
        lastTelemetrySampleMs = now
        val maxSamples = 40

        if (vramTotalGB > 0.0) {
            val pct = (vramUsedGB / vramTotalGB).toFloat().coerceIn(0f, 1f)
            if (vramHistoryPct.isEmpty()) repeat(12) { vramHistoryPct.add(pct) }
            else {
                vramHistoryPct.add(pct)
                if (vramHistoryPct.size > maxSamples) vramHistoryPct.removeAt(0)
            }
        }
        if (ramTotalGB > 0.0) {
            val pct = (ramUsedGB / ramTotalGB).toFloat().coerceIn(0f, 1f)
            if (ramHistoryPct.isEmpty()) repeat(12) { ramHistoryPct.add(pct) }
            else {
                ramHistoryPct.add(pct)
                if (ramHistoryPct.size > maxSamples) ramHistoryPct.removeAt(0)
            }
        }
        if (diskTotalGB > 0.0) {
            val pct = (diskUsedGB / diskTotalGB).toFloat().coerceIn(0f, 1f)
            if (diskHistoryPct.isEmpty()) repeat(12) { diskHistoryPct.add(pct) }
            else {
                diskHistoryPct.add(pct)
                if (diskHistoryPct.size > maxSamples) diskHistoryPct.removeAt(0)
            }
        }
    }

    internal fun applyLiveTelemetryObject(snap: JSONObject) {
        snap.optStringOrNull("gpu_name")?.takeIf { it.isNotEmpty() }?.let { gpuName = stripEmoji(it) }
        snap.optDoubleOrNull("vram_free_gb")?.let { vramFreeGB = it }
        snap.optDoubleOrNull("vram_total_gb")?.let { vramTotalGB = it }
        snap.optDoubleOrNull("vram_used_gb")?.let { vramUsedGB = it }
        snap.optDoubleOrNull("cu_burn_rate_per_hour")?.let { cuBurnRatePerHour = it }
        snap.optDoubleOrNull("ram_total_gb")?.takeIf { it > 0 }?.let { ramTotalGB = it }
        snap.optDoubleOrNull("ram_used_gb")?.let { ramUsedGB = it }
        snap.optDoubleOrNull("ram_free_gb")?.let { ramFreeGB = it }
        snap.optDoubleOrNull("disk_total_gb")?.takeIf { it > 0 }?.let { diskTotalGB = it }
        snap.optDoubleOrNull("disk_used_gb")?.let { diskUsedGB = it }
        snap.optDoubleOrNull("disk_free_gb")?.let { diskFreeGB = it }
        snap.optDoubleOrNull("drive_total_gb")?.takeIf { it > 0 }?.let { driveTotalGB = it }
        snap.optDoubleOrNull("drive_used_gb")?.let { driveUsedGB = it }
        snap.optDoubleOrNull("drive_free_gb")?.let { driveFreeGB = it }
        recordTelemetrySample()
    }

    internal fun resetStaleRemoteRuntimeState(reason: String) {
        stopLiveGenerationTimer()
        isConnected = false
        modelLoaded = false
        loadedModelId = null
        loadedModelIds = emptyList()
        loadedModelsDetail = emptyList()
        isGenerating = false
        isLoadingModel = false
        isDownloadingWeights = false
        isUnloadingModel = false
        comfyRunning = false
        comfyStarting = false
        comfyDownloadingWeight = false
        comfyInstallingPackId = ""
        comfyPublicURL = ""
        progressValue = 0f
        modelTaskProgress = 0f
        vramUsedGB = 0.0; vramFreeGB = 0.0; vramTotalGB = 0.0
        ramTotalGB = 0.0; ramUsedGB = 0.0; ramFreeGB = 0.0
        diskTotalGB = 0.0; diskUsedGB = 0.0; diskFreeGB = 0.0
        vramHistoryPct.clear(); ramHistoryPct.clear(); diskHistoryPct.clear()
        lastTelemetrySampleMs = 0L
        cuBurnRatePerHour = 0.0
        remoteContextSynced = false; remoteIdleSynced = false; remoteApiKeySynced = false
        gpuName = L("未连接 Colab 服务端", "Colab Server Disconnected")
        if (reason.isNotEmpty()) statusMessage = stripEmoji(reason)
    }

    fun switchSelectedModel(spec: OpenModelSpec) {
        if (selectedModelId == spec.id) return
        selectedModelId = spec.id
        taskModeId = spec.modes[0].id
        inferenceSteps = spec.defaultSteps
        modelFiles = spec.defaultFiles
        downloadedCount = 0
        totalFileCount = spec.defaultFiles.size
        allDownloaded = false
        totalDiskSizeGB = 0.0
        modelLoaded = loadedModelIds.contains(spec.id) || (loadedModelId == spec.id)

        PRESET_TEMPLATES.firstOrNull { it.category == spec.modality }?.let { sample ->
            visualPrompt = sample.visual
            soundscapePrompt = sample.soundscape
            musicPrompt = sample.music
        }

        if (spec.modality != ModelModality.COMFY_UI && isConnected) {
            viewModelScope.launch { notifyRemoteSelectModel(spec.id) }
        }
    }


    internal suspend fun postJsonForStatus(url: String, bodyObj: JSONObject, timeoutSec: Long = 30L): JSONObject? =
        withContext(Dispatchers.IO) {
            val client = if (timeoutSec == 30L) apiClient else apiClient.newBuilder()
                .readTimeout(timeoutSec, TimeUnit.SECONDS)
                .build()
            val req = Request.Builder()
                .url(url)
                .post(bodyObj.toString().toRequestBody("application/json".toMediaType()))
                .build()
            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) return@withContext null
                val txt = resp.body?.string().orEmpty()
                if (txt.isEmpty()) null else JSONObject(txt)
            }
        }

    internal suspend fun notifyRemoteSelectModel(modelId: String) {
        val base = normalizedBaseURL.takeIf { it.isNotEmpty() } ?: return
        try {
            val json = postJsonForStatus("$base/minimax_api/select_model", JSONObject().put("model_id", modelId))
            if (json != null) applyStatusJson(json)
        } catch (_: Exception) {
        }
    }

    internal fun applyStatusJson(json: JSONObject) {
        isConnected = true
        gpuName = stripEmoji(json.optStringOrNull("gpu_name") ?: "Colab GPU")
        vramFreeGB = json.optDoubleOrNull("vram_free_gb") ?: 0.0
        vramTotalGB = json.optDoubleOrNull("vram_total_gb") ?: 0.0
        vramUsedGB = json.optDoubleOrNull("vram_used_gb") ?: 0.0
        json.optDoubleOrNull("ram_total_gb")?.let { ramTotalGB = it }
        json.optDoubleOrNull("ram_used_gb")?.let { ramUsedGB = it }
        json.optDoubleOrNull("ram_free_gb")?.let { ramFreeGB = it }
        json.optDoubleOrNull("disk_total_gb")?.let { diskTotalGB = it }
        json.optDoubleOrNull("disk_used_gb")?.let { diskUsedGB = it }
        json.optDoubleOrNull("disk_free_gb")?.let { diskFreeGB = it }
        json.optDoubleOrNull("drive_total_gb")?.let { driveTotalGB = it }
        json.optDoubleOrNull("drive_used_gb")?.let { driveUsedGB = it }
        json.optDoubleOrNull("drive_free_gb")?.let { driveFreeGB = it }
        recordTelemetrySample()

        json.optJSONArray("custom_models")?.let { cmArr ->
            mergeRemoteCustomModels(cmArr)
        }

        loadedModelId = json.optStringOrNull("loaded_model_id")
        val idsArr = json.optJSONArray("loaded_model_ids")
        if (idsArr != null) {
            loadedModelIds = idsArr.toStringList()
        } else if (loadedModelId != null && json.optBoolean("model_loaded", false)) {
            loadedModelIds = listOf(loadedModelId!!)
        } else if (json.optBooleanOrNull("any_model_loaded") == false) {
            loadedModelIds = emptyList()
        }

        val detailsArr = json.optJSONArray("loaded_models_detail")
        if (detailsArr != null) {
            loadedModelsDetail = (0 until detailsArr.length()).mapNotNull { i ->
                val d = detailsArr.optJSONObject(i) ?: return@mapNotNull null
                LoadedModelSummary(
                    id = d.optString("id", ""),
                    name = stripEmoji(d.optString("name", "")),
                    mode = d.optStringOrNull("mode"),
                    vramGb = d.optDoubleOrNull("vram_gb"),
                    contextLength = d.optIntOrNull("context_length")
                )
            }
        } else if (json.optBooleanOrNull("any_model_loaded") == false) {
            loadedModelsDetail = emptyList()
        }

        modelLoaded = loadedModelIds.contains(selectedModelId) || json.optBoolean("model_loaded", false)
        currentModelName = json.optStringOrNull("current_model") ?: L("未加载", "Not Loaded")
        loadedContextLengthTokens = json.optInt("loaded_context_length", 0)

        if (!remoteContextSynced) {
            remoteContextSynced = true
            val rCtx = json.optIntOrNull("context_length")
            if (rCtx != null && rCtx != contextLengthTokens) {
                viewModelScope.launch { syncContextLengthToColab() }
            }
        }

        json.optJSONArray("files")?.takeIf { it.length() > 0 }?.let { fArr ->
            modelFiles = (0 until fArr.length()).mapNotNull { i ->
                val f = fArr.optJSONObject(i) ?: return@mapNotNull null
                ModelFileItem(
                    name = f.optString("name", ""),
                    desc = f.optStringOrNull("desc"),
                    exists = f.optBoolean("exists", false),
                    sizeGb = f.optDouble("size_gb", 0.0)
                )
            }
        }
        downloadedCount = json.optInt("downloaded_count", downloadedCount)
        totalFileCount = json.optInt("total_count", activeModelSpec.defaultFiles.size)
        allDownloaded = json.optBoolean("all_downloaded", downloadedCount == totalFileCount && totalFileCount > 0)
        json.optDoubleOrNull("total_size_gb")?.let { totalDiskSizeGB = it }
        drivePersisted = json.optBoolean("drive_persisted", false)
        isDownloadingWeights = json.optBoolean("is_downloading", false)
        isLoadingModel = json.optBoolean("is_loading_vram", false)
        json.optDoubleOrNull("task_progress")?.let { modelTaskProgress = it.toFloat() }

        uptimeSeconds = json.optInt("uptime_seconds", uptimeSeconds)
        json.optDoubleOrNull("cu_burn_rate_per_hour")?.let { cuBurnRatePerHour = it }
        json.optDoubleOrNull("session_cu_used")?.let { sessionCUUsed = it }
        idleRemainingSeconds = json.optInt("idle_remaining_seconds", idleRemainingSeconds)

        if (!remoteIdleSynced) {
            remoteIdleSynced = true
            val rIdle = json.optIntOrNull("idle_timeout_minutes")
            if (rIdle != null && rIdle != idleTimeoutMinutes) {
                viewModelScope.launch { updateIdleTimeoutConfig(idleTimeoutMinutes) }
            }
        }

        apiRequestCount = json.optInt("api_request_count", apiRequestCount)
        if (apiKeyEnabled && !json.optBoolean("api_key_enabled", false) && !remoteApiKeySynced) {
            remoteApiKeySynced = true
            viewModelScope.launch { syncApiKeyToColab() }
        }

        json.optJSONArray("console_lines")?.takeIf { it.length() > 0 }?.let { cArr ->
            val parsed = cArr.toStringList().map(::stripEmoji)
            liveConsoleLines.clear()
            liveConsoleLines.addAll(parsed)
        }

        json.optJSONObject("comfyui")?.let { cf ->
            comfyRunning = cf.optBoolean("is_running", false)
            comfyStarting = cf.optBoolean("is_starting", false)
            comfySyncingDrive = cf.optBoolean("is_syncing_drive", false)
            comfyDownloadingWeight = cf.optBoolean("is_downloading_weight", false)
            comfyInstallingPackId = cf.optString("installing_pack_id", "")
            comfyProgress = cf.optDouble("progress", 0.0).toFloat()
            cf.optStringOrNull("status_message")?.takeIf { it.isNotEmpty() }?.let { comfyStatusMessage = stripEmoji(it) }
            comfyPublicURL = cf.optString("public_url", "")
            comfyQueueRunning = cf.optInt("queue_running", 0)
            comfyQueuePending = cf.optInt("queue_pending", 0)
            cf.optStringOrNull("persistence_label")?.takeIf { it.isNotEmpty() }?.let { comfyPersistenceLabel = it }
            comfyCustomNodesCount = cf.optInt("custom_nodes_count", 0)
            comfyModelsCount = cf.optInt("models_count", 0)
            comfyWorkflowsCount = cf.optInt("workflows_count", 0)

            cf.optJSONArray("recent_outputs")?.let { outArr ->
                comfyRecentOutputs = (0 until outArr.length()).mapNotNull { i ->
                    val o = outArr.optJSONObject(i) ?: return@mapNotNull null
                    ComfyOutputMediaItem(
                        relPath = o.optString("rel_path", ""),
                        filename = o.optString("filename", ""),
                        mediaType = o.optStringOrNull("media_type"),
                        sizeMb = o.optDoubleOrNull("size_mb")
                    )
                }
            }
            cf.optJSONArray("preset_packs")?.takeIf { it.length() > 0 }?.let { pArr ->
                comfyPresetPacks = (0 until pArr.length()).mapNotNull { i ->
                    val p = pArr.optJSONObject(i) ?: return@mapNotNull null
                    ComfyPresetPackItem(
                        id = p.optString("id", ""),
                        title = p.optString("title", ""),
                        badge = p.optString("badge", ""),
                        scenario = p.optString("scenario", ""),
                        approxSizeGb = p.optDouble("approx_size_gb", 0.0),
                        installedModels = p.optInt("installed_models", 0),
                        totalModels = p.optInt("total_models", 0),
                        ready = p.optBoolean("ready", false)
                    )
                }
            }
        }
    }

    suspend fun checkColabStatus(silent: Boolean = false) {
        val base = normalizedBaseURL
        if (base.isEmpty()) {
            if (!silent) {
                statusMessage = L("请先绑定账号一键唤醒，或填入 gradio.live 服务地址。", "Bind Google account to wake G4, or enter a gradio.live URL.")
            }
            isConnected = false
            return
        }
        if (!silent) {
            statusMessage = L("正在连接并同步云端 GPU 状态...", "Connecting & syncing cloud GPU status...")
        }
        try {
            val (code, bodyStr) = withContext(Dispatchers.IO) {
                val req = Request.Builder()
                    .url("$base/minimax_api/status?model_id=$selectedModelId")
                    .get()
                    .build()
                apiClient.newCall(req).execute().use { resp ->
                    resp.code to resp.body?.string().orEmpty()
                }
            }
            if (code != 200) {
                consecutiveStatusFailures += 1
                if (!silent || consecutiveStatusFailures >= 6) {
                    resetStaleRemoteRuntimeState(L("云端连接不可用 (HTTP $code)，实例可能已休眠或释放。", "Cloud runtime unreachable (HTTP $code)."))
                }
                return
            }
            consecutiveStatusFailures = 0
            val json = JSONObject(bodyStr)
            applyStatusJson(json)
            if (!silent) {
                showToast(L("已连接 $gpuName", "Connected to $gpuName"))
                statusMessage = if (isSelectedModelLoadedInVRAM) {
                    L("已连接 $gpuName · ${activeModelSpec.name} 已驻留显存！", "Connected to $gpuName · ${activeModelSpec.name} ready in VRAM!")
                } else {
                    L("已连接 $gpuName · 可下载并加载模型至显存。", "Connected to $gpuName · Ready to download & load models.")
                }
            }
        } catch (e: Exception) {
            consecutiveStatusFailures += 1
            if (!silent) {
                resetStaleRemoteRuntimeState(L("连接失败: ${e.message}", "Connection failed: ${e.message}"))
            } else if (consecutiveStatusFailures >= 6 && isConnected) {
                resetStaleRemoteRuntimeState(L("与云端连接中断，可点「连接」重试。", "Lost connection to cloud server."))
            }
        }
    }

    fun triggerCheckStatus() {
        viewModelScope.launch { checkColabStatus(silent = false) }
    }

    internal fun startBackgroundTelemetryLoop() {
        if (heartbeatJob?.isActive == true) return
        heartbeatJob = viewModelScope.launch {
            if (normalizedBaseURL.isNotEmpty()) {
                checkColabStatus(silent = true)
            }
            var tick = 0
            while (isActive) {
                delay(3000)
                tick += 1
                if (isConnected) {
                    if (!isGenerating || tick % 5 == 0) {
                        checkColabStatus(silent = true)
                    }
                } else if (normalizedBaseURL.isNotEmpty() && tick % 4 == 0) {
                    checkColabStatus(silent = true)
                }

                if (tick % 16 == 0 && cliLoggedIn && !isWakingColab) {
                    val shouldKeepalive = isConnected && (
                        isGenerating || isDownloadingWeights || isLoadingModel || comfyRunning || comfyStarting ||
                            comfyQueueRunning > 0 || idleTimeoutMinutes <= 0 || idleRemainingSeconds > 75
                        )
                    if (shouldKeepalive) {
                        val st = bridge.keepalive()
                        st.paidBalance?.let { officialCUBalance = it }
                        st.burnRateHourly?.let { officialBurnRate = it }
                        activeVMCount = st.activeAssignments
                    } else {
                        refreshOfficialColabAccountUsage()
                    }
                }
            }
        }
    }

    fun startRemoteDownloadWeights() {
        val base = normalizedBaseURL.takeIf { it.isNotEmpty() } ?: return
        val targetModel = selectedModelId
        viewModelScope.launch {
            isDownloadingWeights = true
            modelTaskProgress = 0.05f
            statusMessage = L("正在启动云端 8 线程高速下载 ${activeModelSpec.name}...", "Starting 8-thread cloud download for ${activeModelSpec.name}...")
            try {
                val curSpec = activeModelSpec
                val payload = JSONObject()
                    .put("model_id", targetModel)
                    .put("hf_token", hfToken)
                    .put("repo_id", if (curSpec.isCustom) curSpec.repoId else "")
                    .put("include_pattern", if (curSpec.isCustom) curSpec.includePattern else "")
                    .put("custom_name", if (curSpec.isCustom) curSpec.nameZh else "")
                postJsonForStatus("$base/minimax_api/download_model", payload)?.let(::applyStatusJson)
                while (isDownloadingWeights && isActive) {
                    delay(1500)
                    val sJson = withContext(Dispatchers.IO) {
                        val req = Request.Builder().url("$base/minimax_api/status?model_id=$targetModel").build()
                        apiClient.newCall(req).execute().use { r ->
                            if (r.isSuccessful) JSONObject(r.body?.string().orEmpty()) else null
                        }
                    }
                    if (sJson != null) {
                        applyStatusJson(sJson)
                        sJson.optStringOrNull("task_message")?.takeIf { it.isNotEmpty() }?.let {
                            statusMessage = localizeServerText(stripEmoji(it))
                        }
                        if (!sJson.optBoolean("is_downloading", false)) {
                            isDownloadingWeights = false
                            break
                        }
                    }
                }
                if (allDownloaded) {
                    showToast(L("权重下载完成！请点「加载到显存」", "Weights Ready! Tap 'Load to VRAM'"))
                }
            } catch (e: Exception) {
                isDownloadingWeights = false
                statusMessage = L("下载权重请求失败: ${e.message}", "Weight download failed: ${e.message}")
            }
        }
    }

    fun loadRemoteModelToVRAM() {
        val base = normalizedBaseURL.takeIf { it.isNotEmpty() } ?: return
        if (!allDownloaded) {
            statusMessage = L("请先点击「下载权重」完成模型拉取。", "Please tap 'Download Weights' first.")
            return
        }
        val targetModel = selectedModelId
        viewModelScope.launch {
            isLoadingModel = true
            modelTaskProgress = 0.15f
            statusMessage = L("正在装载 ${activeModelSpec.name} 至云端 GPU 显存...", "Loading ${activeModelSpec.name} into cloud GPU VRAM...")
            try {
                val curSpec = activeModelSpec
                val payload = JSONObject()
                    .put("model_id", targetModel)
                    .put("precision_mode", "bf16")
                    .put("offload_mode", selectedOffload)
                    .put("context_length", contextLengthTokens)
                    .put("hf_token", hfToken)
                    .put("repo_id", if (curSpec.isCustom) curSpec.repoId else "")
                    .put("include_pattern", if (curSpec.isCustom) curSpec.includePattern else "")
                    .put("custom_name", if (curSpec.isCustom) curSpec.nameZh else "")
                postJsonForStatus("$base/minimax_api/load_model", payload)?.let(::applyStatusJson)
                while (isLoadingModel && isActive) {
                    delay(1500)
                    val sJson = withContext(Dispatchers.IO) {
                        val req = Request.Builder().url("$base/minimax_api/status?model_id=$targetModel").build()
                        apiClient.newCall(req).execute().use { r ->
                            if (r.isSuccessful) JSONObject(r.body?.string().orEmpty()) else null
                        }
                    }
                    if (sJson != null) {
                        applyStatusJson(sJson)
                        sJson.optStringOrNull("task_message")?.takeIf { it.isNotEmpty() }?.let {
                            statusMessage = localizeServerText(stripEmoji(it))
                        }
                        if (!sJson.optBoolean("is_loading_vram", false)) {
                            isLoadingModel = false
                            break
                        }
                    }
                }
                if (isSelectedModelLoadedInVRAM) {
                    showToast(L("${activeModelSpec.shortTitle} 已加载至显存！", "${activeModelSpec.shortTitle} Loaded in VRAM!"))
                }
            } catch (e: Exception) {
                isLoadingModel = false
                statusMessage = L("加载模型失败: ${e.message}", "Load model failed: ${e.message}")
            }
        }
    }

    fun unloadRemoteModelFromVRAM(modelId: String) {
        val base = normalizedBaseURL.takeIf { it.isNotEmpty() } ?: return
        viewModelScope.launch {
            isUnloadingModel = true
            try {
                val json = postJsonForStatus("$base/minimax_api/unload_model", JSONObject().put("model_id", modelId))
                if (json != null) {
                    applyStatusJson(json)
                    showToast(
                        L(
                            "已释放显存 · 可用 ${String.format(Locale.US, "%.1f", vramFreeGB)} GB",
                            "VRAM Freed · ${String.format(Locale.US, "%.1f", vramFreeGB)} GB Free"
                        )
                    )
                }
            } catch (_: Exception) {
            } finally {
                isUnloadingModel = false
            }
        }
    }

    fun setIdleTimeout(minutes: Int) {
        viewModelScope.launch { updateIdleTimeoutConfig(minutes) }
    }

    internal suspend fun updateIdleTimeoutConfig(minutes: Int) {
        idleTimeoutMinutes = minutes
        prefs.edit().putInt("idleTimeoutMinutes", minutes).apply()
        val base = normalizedBaseURL.takeIf { it.isNotEmpty() && isConnected } ?: return
        try {
            postJsonForStatus("$base/minimax_api/config_idle", JSONObject().put("idle_timeout_minutes", minutes))
                ?.let(::applyStatusJson)
        } catch (_: Exception) {
        }
    }

    fun syncContextLengthNow() {
        viewModelScope.launch { syncContextLengthToColab() }
    }

    internal suspend fun syncContextLengthToColab() {
        val base = normalizedBaseURL.takeIf { it.isNotEmpty() && isConnected } ?: return
        try {
            postJsonForStatus("$base/minimax_api/config_context", JSONObject().put("context_length", contextLengthTokens))
                ?.let(::applyStatusJson)
        } catch (_: Exception) {
        }
    }

    fun regenerateApiKey() {
        val gen = "sk-colab-" + UUID.randomUUID().toString().replace("-", "").take(20).lowercase(Locale.US)
        updateApiAccessKey(gen)
        viewModelScope.launch { syncApiKeyToColab() }
    }

    suspend fun syncApiKeyToColab() {
        val base = normalizedBaseURL.takeIf { it.isNotEmpty() && isConnected } ?: return
        val key = if (apiKeyEnabled) apiAccessKey.trim() else ""
        try {
            postJsonForStatus("$base/minimax_api/config_api_key", JSONObject().put("api_key", key))?.let {
                remoteApiKeySynced = true
                applyStatusJson(it)
            }
        } catch (_: Exception) {
        }
    }


    // MARK: - Multimodal Inference & Job Polling

    internal fun startLiveGenerationTimer() {
        generationTimerJob?.cancel()
        generationStartedAtMs = System.currentTimeMillis()
        currentJobElapsed = 0.0
        generationTimerJob = viewModelScope.launch {
            while (isActive && isGenerating) {
                delay(250)
                val el = (System.currentTimeMillis() - generationStartedAtMs) / 1000.0
                if (el > currentJobElapsed) currentJobElapsed = el
            }
        }
    }

    internal fun stopLiveGenerationTimer() {
        generationTimerJob?.cancel()
        generationTimerJob = null
    }

    fun cancelCurrentGeneration() {
        activeGenerationToken = UUID.randomUUID().toString()
        pendingJobId = null
        stopLiveGenerationTimer()
        isGenerating = false
        progressValue = 0f
        statusMessage = L("正在中断云端 GPU 推理任务...", "Aborting cloud GPU inference...")
        val base = normalizedBaseURL.takeIf { it.isNotEmpty() } ?: return
        viewModelScope.launch {
            try {
                postJsonForStatus("$base/minimax_api/cancel", JSONObject(), timeoutSec = 12L)?.let(::applyStatusJson)
            } catch (_: Exception) {
            }
            statusMessage = L("已停止当前生成任务。", "Stopped current generation job.")
        }
    }

    internal fun bitmapToBase64Jpeg(bmp: Bitmap): String {
        val baos = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 90, baos)
        return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
    }

    fun startGeneration() {
        val promptTrimmed = visualPrompt.trim()
        if (promptTrimmed.isEmpty()) {
            statusMessage = L("请先输入提示词或对话内容！", "Please enter a prompt first!")
            return
        }
        if ((taskModeId == "fl2va" || taskModeId == "i2i_edit") && firstFrameBitmap == null && lastFrameBitmap == null) {
            statusMessage = L("请先从手机相册选择至少 1 张参考图片。", "Please select at least 1 reference photo first.")
            return
        }

        viewModelScope.launch {
            checkColabStatus(silent = true)
            if (!isConnected) {
                statusMessage = L("尚未连接云端 Colab，请先前往「云算力」页唤醒或连接。", "Not connected to Colab. Wake or connect in Cloud tab first.")
                return@launch
            }
            if (!isSelectedModelLoadedInVRAM) {
                statusMessage = if (allDownloaded) {
                    L("请先点击上方「加载到显存」装载模型。", "Tap 'Load to VRAM' above first.")
                } else {
                    L("请先点击「下载权重」并「加载到显存」。", "Tap 'Download Weights' and 'Load to VRAM' first.")
                }
                return@launch
            }

            val base = normalizedBaseURL
            val myToken = UUID.randomUUID().toString()
            activeGenerationToken = myToken
            isGenerating = true
            progressValue = 0f
            currentJobElapsed = 0.0
            startLiveGenerationTimer()

            val resOpt = if (activeModelSpec.modality == ModelModality.IMAGE) selectedImageResolution else selectedVideoResolution
            val chatPayload = JSONArray()
            var assistantIdx: Int? = null

            if (activeModelSpec.modality == ModelModality.LLM_VLM) {
                val hasImg = (taskModeId == "vlm" && (firstFrameBitmap != null || lastFrameBitmap != null))
                chatHistory.add(ChatTurnItem(role = "user", content = promptTrimmed, hasImage = hasImg))
                for (turn in chatHistory) {
                    chatPayload.put(JSONObject().put("role", turn.role).put("content", turn.content))
                }
                chatHistory.add(ChatTurnItem(role = "assistant", content = "", thinking = ""))
                assistantIdx = chatHistory.size - 1
                pendingAssistantIndex = assistantIdx
                visualPrompt = ""
            } else {
                pendingAssistantIndex = null
            }

            try {
                val payload = withContext(Dispatchers.Default) {
                    val obj = JSONObject()
                        .put("model_id", selectedModelId)
                        .put("task_mode", taskModeId)
                        .put("visual_prompt", promptTrimmed)
                        .put("negative_prompt", negativePrompt)
                        .put("soundscape_prompt", soundscapePrompt)
                        .put("music_prompt", musicPrompt)
                        .put("width", resOpt.width)
                        .put("height", resOpt.height)
                        .put("num_frames", numFrames.toInt())
                        .put("num_inference_steps", inferenceSteps.toInt())
                        .put("fps", fps.toInt())
                        .put("seed", seed)
                        .put("randomize_seed", randomizeSeed)
                        .put("batch_count", if (activeModelSpec.modality == ModelModality.LLM_VLM) 1 else batchCount)
                        .put("denoising_strength", denoisingStrength.toDouble())
                        .put("enable_thinking", enableThinking)
                        .put("max_new_tokens", maxNewTokens.toInt())
                        .put("context_length", contextLengthTokens)
                        .put("temperature", temperature.toDouble())
                        .put("top_p", topP.toDouble())
                        .put("top_k", 20)
                        .put("min_p", 0.0)
                        .put("presence_penalty", presencePenalty.toDouble())
                        .put("repeat_penalty", 1.0)
                        .put("reasoning_effort", reasoningEffort)

                    if (chatPayload.length() > 0) obj.put("chat_messages", chatPayload)
                    if (taskModeId in listOf("fl2va", "i2v", "i2i_edit", "vlm")) {
                        firstFrameBitmap?.let { obj.put("first_frame_base64", bitmapToBase64Jpeg(it)) }
                        lastFrameBitmap?.let { obj.put("last_frame_base64", bitmapToBase64Jpeg(it)) }
                    }
                    obj
                }

                val jobJson = withContext(Dispatchers.IO) {
                    val req = Request.Builder()
                        .url("$base/minimax_api/generate")
                        .post(payload.toString().toRequestBody("application/json".toMediaType()))
                        .build()
                    apiClient.newCall(req).execute().use { resp ->
                        JSONObject(resp.body?.string().orEmpty())
                    }
                }
                if (activeGenerationToken != myToken) return@launch

                val st = jobJson.optString("status", "")
                val msg = jobJson.optString("message", "")
                val jobId = jobJson.optString("job_id", "")

                if (st == "failed" || jobId.isEmpty()) {
                    stopLiveGenerationTimer()
                    isGenerating = false
                    statusMessage = localizeServerText(stripEmoji(msg))
                    if (assistantIdx != null && assistantIdx < chatHistory.size) {
                        chatHistory[assistantIdx] = chatHistory[assistantIdx].copy(content = localizeServerText(stripEmoji(msg)))
                    }
                    return@launch
                }

                pendingJobId = jobId
                pollColabJob(base, jobId, assistantIdx, myToken)
            } catch (e: Exception) {
                if (activeGenerationToken != myToken) return@launch
                stopLiveGenerationTimer()
                isGenerating = false
                statusMessage = L("提交生成任务失败: ${e.message}", "Failed to submit job: ${e.message}")
            }
        }
    }

    internal suspend fun pollColabJob(baseURL: String, jobId: String, assistantRowIndex: Int?, token: String) {
        val pollUrl = "$baseURL/minimax_api/job/$jobId"
        val downloadedFilenames = mutableSetOf<String>()
        var consecutiveErrors = 0
        val intervalMs = if (assistantRowIndex != null) 650L else 1000L

        while (isGenerating && activeGenerationToken == token) {
            delay(intervalMs)
            if (!isGenerating || activeGenerationToken != token) break
            try {
                val (code, bodyStr) = withContext(Dispatchers.IO) {
                    val req = Request.Builder().url(pollUrl).get().build()
                    apiClient.newCall(req).execute().use { resp ->
                        resp.code to resp.body?.string().orEmpty()
                    }
                }
                if (!isGenerating || activeGenerationToken != token) break

                if (code == 404) {
                    consecutiveErrors += 2
                    if (consecutiveErrors >= 8) {
                        pendingJobId = null
                        isGenerating = false
                        stopLiveGenerationTimer()
                        break
                    }
                    continue
                }
                if (code != 200) {
                    consecutiveErrors += 1
                    if (consecutiveErrors >= 14) {
                        isGenerating = false
                        stopLiveGenerationTimer()
                        break
                    }
                    continue
                }
                consecutiveErrors = 0
                val job = JSONObject(bodyStr)
                job.optJSONObject("telemetry")?.let(::applyLiveTelemetryObject)

                val st = job.optString("status", "")
                if (st != "completed") {
                    progressValue = job.optDouble("progress", 0.0).toFloat().coerceIn(0f, 0.98f)
                }
                statusMessage = localizeServerText(stripEmoji(job.optString("message", "")))
                job.optDoubleOrNull("elapsed")?.takeIf { it >= 0 }?.let {
                    if (it > currentJobElapsed) currentJobElapsed = it
                }
                job.optIntOrNull("seed")?.let { seed = it }
                job.optJSONArray("console_lines")?.takeIf { it.length() > 0 }?.let { cArr ->
                    val parsed = cArr.toStringList().map(::stripEmoji)
                    liveConsoleLines.clear()
                    liveConsoleLines.addAll(parsed)
                }

                if (assistantRowIndex != null && assistantRowIndex < chatHistory.size) {
                    val prev = chatHistory[assistantRowIndex]
                    val newThink = job.optStringOrNull("thinking_output") ?: prev.thinking
                    val newAns = job.optStringOrNull("text_output") ?: prev.content
                    if (newThink != prev.thinking || newAns != prev.content) {
                        chatHistory[assistantRowIndex] = prev.copy(thinking = newThink, content = newAns)
                    }
                }

                job.optJSONArray("completed_videos")?.let { items ->
                    for (i in 0 until items.length()) {
                        val item = items.optJSONObject(i) ?: continue
                        val fn = item.optString("filename", "")
                        val vUrl = item.optString("video_url", "")
                        if (fn.isNotEmpty() && vUrl.isNotEmpty() && !downloadedFilenames.contains(fn)) {
                            downloadedFilenames.add(fn)
                            val itemEl = item.optDoubleOrNull("elapsed") ?: job.optDoubleOrNull("elapsed")
                            downloadMediaToLocal("$baseURL$vUrl", fn, itemEl)
                        }
                    }
                }

                if (st == "completed") {
                    val finalEl = job.optDoubleOrNull("elapsed") ?: currentJobElapsed
                    lastCompletedElapsed = finalEl
                    val fname = job.optStringOrNull("filename")
                    val vPath = job.optStringOrNull("video_url")
                    if (!fname.isNullOrEmpty() && !vPath.isNullOrEmpty() && !downloadedFilenames.contains(fname)) {
                        downloadedFilenames.add(fname)
                        progressValue = 0.99f
                        downloadMediaToLocal("$baseURL$vPath", fname, finalEl)
                    }
                    pendingJobId = null
                    stopLiveGenerationTimer()
                    isGenerating = false
                    progressValue = 1.0f
                    checkColabStatus(silent = true)
                    break
                } else if (st == "failed") {
                    if (assistantRowIndex != null && assistantRowIndex < chatHistory.size && chatHistory[assistantRowIndex].content.isEmpty()) {
                        chatHistory[assistantRowIndex] = chatHistory[assistantRowIndex].copy(
                            content = localizeServerText(stripEmoji(job.optString("message", "")))
                        )
                    }
                    pendingJobId = null
                    stopLiveGenerationTimer()
                    isGenerating = false
                    progressValue = 0f
                    break
                }
            } catch (_: Exception) {
                consecutiveErrors += 1
                if (consecutiveErrors >= 20) {
                    stopLiveGenerationTimer()
                    isGenerating = false
                    break
                }
            }
        }
    }

}
