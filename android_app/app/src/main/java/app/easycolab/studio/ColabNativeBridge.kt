package app.easycolab.studio

import android.content.Context
import android.net.Uri
import android.util.Base64
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import okhttp3.FormBody
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.security.SecureRandom
import java.time.Instant
import java.util.Locale
import java.util.UUID
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

data class NativeAuthStatus(
    val ok: Boolean,
    val loggedIn: Boolean,
    val email: String,
    val paidBalance: Double? = null,
    val burnRateHourly: Double? = null,
    val activeAssignments: Int = 0,
    val message: String? = null,
    val error: String? = null
)

data class NativeOAuthSessionRequest(
    val authUrl: String,
    val callbackScheme: String?,
    val redirectUri: String
)

private data class StoredOAuthCredentials(
    var accessToken: String,
    var refreshToken: String,
    var expiryTimestamp: Double,
    var clientId: String,
    var clientSecret: String,
    var scopes: List<String>,
    var email: String
)

private data class RuntimeSessionRecord(
    val name: String,
    val token: String,
    val url: String,
    val endpoint: String,
    val accelerator: String,
    val kernelId: String? = null,
    val sessionId: String? = null
)

class ColabBridgeException(message: String, val httpCode: Int = 0) : Exception(message)

class ColabNativeBridge private constructor(context: Context) {

    companion object {
        @Volatile
        private var INSTANCE: ColabNativeBridge? = null

        fun getInstance(context: Context): ColabNativeBridge =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: ColabNativeBridge(context.applicationContext).also { INSTANCE = it }
            }

        private val encodedClientIdBytes = intArrayOf(
            109, 108, 110, 106, 98, 108, 106, 111, 107, 98, 111, 106, 119, 108, 43, 40, 110, 42,
            108, 61, 42, 51, 108, 50, 52, 111, 106, 108, 42, 46, 98, 63, 48, 47, 43, 98, 105, 62,
            51, 105, 110, 107, 50, 47, 40, 116, 59, 42, 42, 41, 116, 61, 53, 53, 61, 54, 63, 47,
            41, 63, 40, 57, 53, 52, 46, 63, 52, 46, 116, 57, 53, 55
        )
        private val encodedClientSecretBytes = intArrayOf(
            62, 119, 28, 22, 99, 111, 11, 107, 99, 43, 109, 23, 11, 55, 28, 42, 62, 109, 50, 18, 30, 106, 14, 35
        )

        private fun decodeXor(bytes: IntArray, key: Int = 0x5A): String =
            String(ByteArray(bytes.size) { i -> (bytes[i] xor key).toByte() }, Charsets.UTF_8)
    }

    private val colabDomain = "https://colab.research.google.com"
    private val remoteRedirectUri = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
    private val publicScopes = listOf(
        "openid",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/colaboratory",
        "https://www.googleapis.com/auth/drive.file"
    )

    private val secureDir = File(context.filesDir, "easycolab_secure").apply { mkdirs() }
    private val credsFile = File(secureDir, "oauth_creds.json")
    private val sessionFile = File(secureDir, "colab_session.json")
    private val customOAuthFile = File(secureDir, "custom_oauth_client.json")
    private val pkceStateFile = File(secureDir, "pkce_state.json")

    private val mutex = Mutex()
    private var pendingVerifier: String? = null
    private var pendingRedirectUri: String? = null

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(25, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    fun hasCustomOAuthConfig(): Boolean = customOAuthFile.exists()

    private fun loadActiveOAuthClient(): Pair<String, String> {
        if (customOAuthFile.exists()) {
            try {
                val obj = JSONObject(customOAuthFile.readText())
                val cid = obj.optString("client_id", "")
                val sec = obj.optString("client_secret", "")
                if (cid.isNotBlank() && sec.isNotBlank()) return cid to sec
            } catch (_: Exception) {
            }
        }
        return decodeXor(encodedClientIdBytes) to decodeXor(encodedClientSecretBytes)
    }

    suspend fun saveCustomOAuthJson(rawJson: String): Boolean = withContext(Dispatchers.IO) {
        val trimmed = rawJson.trim()
        if (trimmed.isEmpty()) {
            customOAuthFile.delete()
            return@withContext false
        }
        val root = JSONObject(trimmed)
        val inner = root.optJSONObject("installed") ?: root.optJSONObject("web") ?: root
        val cid = inner.optString("client_id", "").trim()
        val sec = inner.optString("client_secret", "").trim()
        if (cid.isEmpty() || sec.isEmpty()) {
            throw ColabBridgeException(L("JSON 中缺少 client_id 或 client_secret", "Missing client_id or client_secret in JSON"))
        }
        val saved = JSONObject().put("client_id", cid).put("client_secret", sec)
        customOAuthFile.writeText(saved.toString())
        true
    }

    private fun saveCredentials(creds: StoredOAuthCredentials) {
        val obj = JSONObject()
            .put("accessToken", creds.accessToken)
            .put("refreshToken", creds.refreshToken)
            .put("expiryTimestamp", creds.expiryTimestamp)
            .put("clientId", creds.clientId)
            .put("clientSecret", creds.clientSecret)
            .put("scopes", JSONArray(creds.scopes))
            .put("email", creds.email)
        credsFile.writeText(obj.toString())
    }

    private fun loadCredentials(): StoredOAuthCredentials? {
        if (!credsFile.exists()) return null
        return try {
            val obj = JSONObject(credsFile.readText())
            StoredOAuthCredentials(
                accessToken = obj.optString("accessToken", ""),
                refreshToken = obj.optString("refreshToken", ""),
                expiryTimestamp = obj.optDouble("expiryTimestamp", 0.0),
                clientId = obj.optString("clientId", ""),
                clientSecret = obj.optString("clientSecret", ""),
                scopes = obj.optJSONArray("scopes")?.toStringList() ?: publicScopes,
                email = obj.optString("email", "")
            )
        } catch (_: Exception) {
            null
        }
    }

    private fun loadSessionRecord(): RuntimeSessionRecord? {
        if (!sessionFile.exists()) return null
        return try {
            val obj = JSONObject(sessionFile.readText())
            RuntimeSessionRecord(
                name = obj.optString("name", "easycolab"),
                token = obj.optString("token", ""),
                url = obj.optString("url", ""),
                endpoint = obj.optString("endpoint", ""),
                accelerator = obj.optString("accelerator", "G4"),
                kernelId = obj.optStringOrNull("kernelId"),
                sessionId = obj.optStringOrNull("sessionId")
            )
        } catch (_: Exception) {
            null
        }
    }

    private fun saveSessionRecord(rec: RuntimeSessionRecord?) {
        if (rec == null) {
            sessionFile.delete()
            return
        }
        val obj = JSONObject()
            .put("name", rec.name)
            .put("token", rec.token)
            .put("url", rec.url)
            .put("endpoint", rec.endpoint)
            .put("accelerator", rec.accelerator)
            .put("kernelId", rec.kernelId ?: JSONObject.NULL)
            .put("sessionId", rec.sessionId ?: JSONObject.NULL)
        sessionFile.writeText(obj.toString())
    }

    suspend fun logout() = withContext(Dispatchers.IO) {
        mutex.withLock {
            credsFile.delete()
            sessionFile.delete()
            pkceStateFile.delete()
            pendingVerifier = null
            pendingRedirectUri = null
        }
    }

    private fun base64UrlNoPad(bytes: ByteArray): String =
        Base64.encodeToString(bytes, Base64.URL_SAFE or Base64.NO_PADDING or Base64.NO_WRAP)

    suspend fun generateAuthSession(preferCustomScheme: Boolean = true): NativeOAuthSessionRequest =
        withContext(Dispatchers.IO) {
            mutex.withLock {
                val rng = SecureRandom()
                val verifierBytes = ByteArray(36).also { rng.nextBytes(it) }
                val verifier = base64UrlNoPad(verifierBytes)
                val challenge = base64UrlNoPad(MessageDigest.getInstance("SHA-256").digest(verifier.toByteArray(Charsets.UTF_8)))
                val stateStr = base64UrlNoPad(ByteArray(18).also { rng.nextBytes(it) })

                val (clientId, _) = loadActiveOAuthClient()
                val suffix = ".apps.googleusercontent.com"
                val callbackScheme: String?
                val redirectUri: String

                if (preferCustomScheme && clientId.endsWith(suffix) && !hasCustomOAuthConfig()) {
                    val prefix = clientId.removeSuffix(suffix)
                    callbackScheme = "com.googleusercontent.apps.$prefix"
                    redirectUri = "$callbackScheme:/oauth2redirect"
                } else {
                    callbackScheme = null
                    redirectUri = remoteRedirectUri
                }

                pendingVerifier = verifier
                pendingRedirectUri = redirectUri
                pkceStateFile.writeText(
                    JSONObject().put("verifier", verifier).put("redirectUri", redirectUri).toString()
                )

                val urlBuilder = "https://accounts.google.com/o/oauth2/v2/auth".toHttpUrl().newBuilder()
                    .addQueryParameter("response_type", "code")
                    .addQueryParameter("client_id", clientId)
                    .addQueryParameter("redirect_uri", redirectUri)
                    .addQueryParameter("scope", publicScopes.joinToString(" "))
                    .addQueryParameter("state", stateStr)
                    .addQueryParameter("code_challenge", challenge)
                    .addQueryParameter("code_challenge_method", "S256")
                    .addQueryParameter("prompt", "select_account consent")
                    .addQueryParameter("access_type", "offline")

                if (callbackScheme == null) {
                    urlBuilder.addQueryParameter("token_usage", "remote")
                }

                NativeOAuthSessionRequest(
                    authUrl = urlBuilder.build().toString(),
                    callbackScheme = callbackScheme,
                    redirectUri = redirectUri
                )
            }
        }

    suspend fun exchangeAuthCode(rawInput: String, overrideRedirectUri: String? = null): NativeAuthStatus =
        withContext(Dispatchers.IO) {
            val trimmed = rawInput.trim()
            if (trimmed.isEmpty()) {
                return@withContext NativeAuthStatus(ok = false, loggedIn = false, email = "", error = "Empty code")
            }

            var code = trimmed
            if (trimmed.contains(":/") || trimmed.contains("code=") || trimmed.contains("error=")) {
                try {
                    val uri = Uri.parse(trimmed)
                    val oauthErr = uri.getQueryParameter("error")
                    if (!oauthErr.isNullOrEmpty()) {
                        return@withContext NativeAuthStatus(ok = false, loggedIn = false, email = "", error = oauthErr)
                    }
                    val extracted = uri.getQueryParameter("code")
                    if (!extracted.isNullOrEmpty()) {
                        code = extracted
                    }
                } catch (_: Exception) {
                }
            }

            var persistedVerifier: String? = null
            var persistedRedirect: String? = null
            if (pkceStateFile.exists()) {
                try {
                    val st = JSONObject(pkceStateFile.readText())
                    persistedVerifier = st.optStringOrNull("verifier")
                    persistedRedirect = st.optStringOrNull("redirectUri")
                } catch (_: Exception) {
                }
            }

            val verifier = pendingVerifier ?: persistedVerifier ?: ""
            val activeRedirectUri = overrideRedirectUri ?: pendingRedirectUri ?: persistedRedirect ?: remoteRedirectUri
            val (clientId, clientSecret) = loadActiveOAuthClient()

            val formBuilder = FormBody.Builder()
                .add("code", code)
                .add("client_id", clientId)
                .add("client_secret", clientSecret)
                .add("redirect_uri", activeRedirectUri)
                .add("grant_type", "authorization_code")
            if (verifier.isNotEmpty()) {
                formBuilder.add("code_verifier", verifier)
            }

            val req = Request.Builder()
                .url("https://oauth2.googleapis.com/token")
                .post(formBuilder.build())
                .build()

            try {
                httpClient.newCall(req).execute().use { resp ->
                    val bodyStr = resp.body?.string().orEmpty()
                    val json = try { JSONObject(bodyStr) } catch (_: Exception) { null }
                    if (!resp.isSuccessful || json == null || !json.has("access_token")) {
                        val errDesc = json?.optStringOrNull("error_description")
                            ?: json?.optStringOrNull("error")
                            ?: "HTTP ${resp.code}"
                        return@withContext NativeAuthStatus(ok = false, loggedIn = false, email = "", error = errDesc)
                    }

                    val accessToken = json.optString("access_token", "")
                    val refreshToken = json.optStringOrNull("refresh_token")?.takeIf { it.isNotEmpty() }
                        ?: loadCredentials()?.refreshToken.orEmpty()
                    val expiresIn = json.optDouble("expires_in", 3500.0)
                    val scopeStr = json.optString("scope", publicScopes.joinToString(" "))
                    val email = fetchUserEmail(accessToken)

                    mutex.withLock {
                        saveSessionRecord(null)
                        pkceStateFile.delete()
                        pendingVerifier = null
                        pendingRedirectUri = null
                        saveCredentials(
                            StoredOAuthCredentials(
                                accessToken = accessToken,
                                refreshToken = refreshToken,
                                expiryTimestamp = System.currentTimeMillis() / 1000.0 + expiresIn - 60.0,
                                clientId = clientId,
                                clientSecret = clientSecret,
                                scopes = scopeStr.split(" ").filter { it.isNotBlank() },
                                email = email
                            )
                        )
                    }
                }
                fetchAuthStatus()
            } catch (e: Exception) {
                NativeAuthStatus(ok = false, loggedIn = false, email = "", error = e.message ?: "Network error")
            }
        }

    private fun fetchUserEmail(accessToken: String): String {
        return try {
            val req = Request.Builder()
                .url("https://www.googleapis.com/oauth2/v2/userinfo")
                .header("Authorization", "Bearer $accessToken")
                .build()
            httpClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) return ""
                JSONObject(resp.body?.string().orEmpty()).optString("email", "").trim()
            }
        } catch (_: Exception) {
            ""
        }
    }

    private suspend fun validAccessToken(): StoredOAuthCredentials = withContext(Dispatchers.IO) {
        mutex.withLock {
            val creds = loadCredentials() ?: throw ColabBridgeException(L("尚未绑定 Google 账号", "Google account not bound"))
            val nowSec = System.currentTimeMillis() / 1000.0
            if (nowSec < creds.expiryTimestamp && creds.accessToken.isNotEmpty()) {
                return@withLock creds
            }
            if (creds.refreshToken.isEmpty()) {
                throw ColabBridgeException(L("登录凭证已过期，请重新绑定账号", "Session expired, please sign in again"))
            }

            val form = FormBody.Builder()
                .add("client_id", creds.clientId)
                .add("client_secret", creds.clientSecret)
                .add("refresh_token", creds.refreshToken)
                .add("grant_type", "refresh_token")
                .build()
            val req = Request.Builder()
                .url("https://oauth2.googleapis.com/token")
                .post(form)
                .build()
            httpClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    throw ColabBridgeException(L("刷新 Google Token 失败", "Failed to refresh Google token"))
                }
                val json = JSONObject(resp.body?.string().orEmpty())
                val newToken = json.optString("access_token", "")
                if (newToken.isEmpty()) throw ColabBridgeException("Empty refreshed token")
                val expiresIn = json.optDouble("expires_in", 3500.0)
                creds.accessToken = newToken
                creds.expiryTimestamp = System.currentTimeMillis() / 1000.0 + expiresIn - 60.0
                if (creds.email.isEmpty()) {
                    creds.email = fetchUserEmail(newToken)
                }
                saveCredentials(creds)
                creds
            }
        }
    }

    private fun stripXssi(text: String): String {
        if (text.startsWith(")]}'\n")) return text.substring(5)
        if (text.startsWith(")]}'")) return text.substring(4).trim()
        return text
    }

    private suspend fun colabTunnelRequest(
        path: String,
        method: String = "GET",
        queryParams: List<Pair<String, String>> = emptyList(),
        extraHeaders: Map<String, String> = emptyMap()
    ): JSONObject = withContext(Dispatchers.IO) {
        val creds = validAccessToken()
        val urlBuilder = (colabDomain + path).toHttpUrl().newBuilder()
        var hasAuthUser = false
        for ((k, v) in queryParams) {
            if (k == "authuser") hasAuthUser = true
            urlBuilder.addQueryParameter(k, v)
        }
        if (!hasAuthUser) urlBuilder.addQueryParameter("authuser", "0")

        val reqBuilder = Request.Builder()
            .url(urlBuilder.build())
            .header("Authorization", "Bearer ${creds.accessToken}")
            .header("Accept", "application/json")
            .header("X-Colab-Client-Agent", "colab-cli")

        for ((k, v) in extraHeaders) reqBuilder.header(k, v)
        if (method == "POST") {
            reqBuilder.post(ByteArray(0).toRequestBody(null))
        } else {
            reqBuilder.get()
        }

        httpClient.newCall(reqBuilder.build()).execute().use { resp ->
            val raw = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) {
                if (resp.code == 412) {
                    throw ColabBridgeException(
                        L("当前账号活跃 Colab 实例已达上限 (HTTP 412)，请先点击下方「断开并释放实例」", "Too many active Colab instances (HTTP 412). Terminate existing VMs first."),
                        412
                    )
                }
                throw ColabBridgeException("HTTP ${resp.code}: ${raw.take(200)}", resp.code)
            }
            val clean = stripXssi(raw).trim()
            if (clean.isEmpty()) JSONObject() else try { JSONObject(clean) } catch (_: Exception) { JSONObject() }
        }
    }

    suspend fun fetchAuthStatus(): NativeAuthStatus = withContext(Dispatchers.IO) {
        val stored = loadCredentials() ?: return@withContext NativeAuthStatus(ok = true, loggedIn = false, email = "")
        try {
            val creds = validAccessToken()
            val ccu = colabTunnelRequest("/tun/m/ccu-info")
            val bal = ccu.optDouble("currentBalance", 0.0)
            val rate = ccu.optDouble("consumptionRateHourly", 0.0)
            val count = ccu.optInt("assignmentsCount", 0)
            NativeAuthStatus(
                ok = true,
                loggedIn = true,
                email = creds.email,
                paidBalance = (bal * 100.0).roundToInt() / 100.0,
                burnRateHourly = (rate * 100.0).roundToInt() / 100.0,
                activeAssignments = count
            )
        } catch (e: ColabBridgeException) {
            if (e.message?.contains("绑定") == true || e.message?.contains("bound") == true) {
                NativeAuthStatus(ok = true, loggedIn = false, email = "")
            } else {
                NativeAuthStatus(ok = false, loggedIn = true, email = stored.email, error = e.message)
            }
        } catch (e: Exception) {
            NativeAuthStatus(ok = false, loggedIn = true, email = stored.email, error = e.message)
        }
    }

    private suspend fun listAssignments(): List<JSONObject> {
        val res = colabTunnelRequest("/tun/m/assignments")
        val arr = res.optJSONArray("assignments") ?: return emptyList()
        return (0 until arr.length()).mapNotNull { arr.optJSONObject(it) }
    }

    private suspend fun unassignEndpoint(endpoint: String) {
        if (endpoint.isEmpty()) return
        try {
            val getResp = colabTunnelRequest("/tun/m/unassign/$endpoint", method = "GET")
            val xsrf = getResp.optString("token", "")
            if (xsrf.isNotEmpty()) {
                colabTunnelRequest(
                    "/tun/m/unassign/$endpoint",
                    method = "POST",
                    extraHeaders = mapOf("X-Goog-Colab-Token" to xsrf)
                )
            }
        } catch (_: Exception) {
        }
    }

    suspend fun stopAll(): NativeAuthStatus = withContext(Dispatchers.IO) {
        if (loadCredentials() == null) {
            return@withContext NativeAuthStatus(ok = false, loggedIn = false, email = "", error = "Not logged in")
        }
        var stoppedCount = 0
        loadSessionRecord()?.let { rec ->
            try {
                executeCodeOnKernel(
                    rec,
                    """
                    import urllib.request
                    for _p in (27860, 7860):
                        try:
                            urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{_p}/minimax_api/comfyui/sync_drive', data=b'{}', headers={'Content-Type':'application/json'}, method='POST'), timeout=12)
                            break
                        except Exception:
                            pass
                    """.trimIndent(),
                    timeoutMs = 15_000L
                )
            } catch (_: Exception) {
            }
            unassignEndpoint(rec.endpoint)
            saveSessionRecord(null)
            stoppedCount += 1
        }

        try {
            for (item in listAssignments()) {
                val ep = item.optString("endpoint", "")
                if (ep.isNotEmpty()) {
                    unassignEndpoint(ep)
                    stoppedCount += 1
                }
            }
        } catch (_: Exception) {
        }

        val st = fetchAuthStatus()
        val rate = st.burnRateHourly ?: 0.0
        st.copy(
            message = L(
                "已彻底释放全部云端实例 ($stoppedCount 台)！当前计费速率: ${String.format(Locale.US, "%.2f", rate)} CU/h",
                "Terminated all cloud instances ($stoppedCount). Current burn rate: ${String.format(Locale.US, "%.2f", rate)} CU/h"
            )
        )
    }

    private fun webSafeNotebookHash(): String {
        val s = UUID.randomUUID().toString().lowercase(Locale.US).replace("-", "_")
        return s + ".".repeat((44 - s.length).coerceAtLeast(0))
    }

    private suspend fun allocateFreshSession(gpuType: String, onLog: (String) -> Unit): RuntimeSessionRecord {
        val accel = listOf("G4", "H100", "A100", "L4", "T4").firstOrNull { it.equals(gpuType, ignoreCase = true) } ?: "G4"
        val useHighMem = accel !in setOf("G4", "H100", "L4")
        val gpuDesc = if (accel == "G4") "G4 (RTX PRO 6000 · 96GB)" else accel
        onLog(L("正在向 Google Colab 申请分配 $gpuDesc 实例...", "Requesting $gpuDesc instance from Google Colab..."))

        suspend fun performAssign(highMem: Boolean): JSONObject {
            val q = mutableListOf(
                "nbh" to webSafeNotebookHash(),
                "variant" to "GPU",
                "accelerator" to accel
            )
            if (highMem) q.add("shape" to "hm")
            val getResp = colabTunnelRequest("/tun/m/assign", method = "GET", queryParams = q)
            if (getResp.has("runtimeProxyInfo")) return getResp
            val xsrf = getResp.optString("token", "")
            if (xsrf.isEmpty()) {
                throw ColabBridgeException(L("未能获取实例分配令牌，请检查账号 CU 余额", "Failed to obtain assignment token. Check CU balance."))
            }
            return colabTunnelRequest(
                "/tun/m/assign",
                method = "POST",
                queryParams = q,
                extraHeaders = mapOf("X-Goog-Colab-Token" to xsrf)
            )
        }

        val resp = try {
            performAssign(useHighMem)
        } catch (e: ColabBridgeException) {
            if (e.httpCode == 400 && useHighMem) performAssign(false) else throw e
        }

        val endpoint = resp.optString("endpoint", "")
        val proxy = resp.optJSONObject("runtimeProxyInfo")
        val proxyToken = proxy?.optString("token", "").orEmpty()
        val proxyUrl = proxy?.optString("url", "").orEmpty()
        if (endpoint.isEmpty() || proxyToken.isEmpty() || proxyUrl.isEmpty()) {
            throw ColabBridgeException(L("分配失败：账号可能无可用付费 CU 或当前 GPU 缺货，可尝试切换 L4 / A100", "Allocation failed: check paid CU balance or try L4 / A100."))
        }

        val rec = RuntimeSessionRecord("easycolab", proxyToken, proxyUrl, endpoint, accel)
        saveSessionRecord(rec)
        onLog(L("成功分配 $accel 云端虚拟机！正在等待容器内核就绪...", "Allocated $accel cloud VM! Waiting for Jupyter kernel..."))
        return rec
    }

    private suspend fun waitForRuntimeReady(session: RuntimeSessionRecord, timeoutMs: Long): Boolean =
        withContext(Dispatchers.IO) {
            val base = session.url.trimEnd('/')
            val url = "$base/api/kernels".toHttpUrl().newBuilder()
                .addQueryParameter("authuser", "0")
                .addQueryParameter("colab-runtime-proxy-token", session.token)
                .build()

            val deadline = System.currentTimeMillis() + timeoutMs
            var notFoundCount = 0
            while (System.currentTimeMillis() < deadline) {
                try {
                    val req = Request.Builder()
                        .url(url)
                        .header("Accept", "application/json")
                        .header("X-Colab-Client-Agent", "colab-cli")
                        .header("X-Colab-Runtime-Proxy-Token", session.token)
                        .header("Authorization", "Bearer ${session.token}")
                        .build()
                    httpClient.newCall(req).execute().use { resp ->
                        if (resp.code == 200) return@withContext true
                        if (resp.code == 404 || resp.code == 410) {
                            notFoundCount++
                            if (notFoundCount >= 2) return@withContext false
                        }
                    }
                } catch (_: Exception) {
                }
                delay(2000)
            }
            false
        }

    private suspend fun uploadFileToColab(session: RuntimeSessionRecord, bytes: ByteArray, remotePath: String) =
        withContext(Dispatchers.IO) {
            val cleanRemote = remotePath.trim('/')
            val base = session.url.trimEnd('/')
            val url = "$base/api/contents/$cleanRemote".toHttpUrl().newBuilder()
                .addQueryParameter("authuser", "0")
                .addQueryParameter("colab-runtime-proxy-token", session.token)
                .build()

            val filename = cleanRemote.substringAfterLast('/')
            val payload = JSONObject()
                .put("name", filename)
                .put("path", cleanRemote)
                .put("type", "file")
                .put("format", "base64")
                .put("content", Base64.encodeToString(bytes, Base64.NO_WRAP))
                .put("chunk", 1)

            val req = Request.Builder()
                .url(url)
                .put(payload.toString().toRequestBody("application/json".toMediaType()))
                .header("Accept", "application/json")
                .header("X-Colab-Client-Agent", "colab-cli")
                .header("X-Colab-Runtime-Proxy-Token", session.token)
                .header("Authorization", "Bearer ${session.token}")
                .build()

            httpClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    throw ColabBridgeException("Upload failed HTTP ${resp.code}: ${resp.body?.string()?.take(180)}", resp.code)
                }
            }
        }

    private suspend fun buildRcloneDriveConf(): String? {
        val creds = try { validAccessToken() } catch (_: Exception) { return null }
        if (creds.refreshToken.isEmpty()) return null
        val driveScope = when {
            creds.scopes.any { it.endsWith("/drive") } -> "drive"
            creds.scopes.any { it.contains("drive.file") } -> "drive.file"
            else -> ""
        }
        if (driveScope.isEmpty()) return null
        val tokJson = JSONObject()
            .put("access_token", creds.accessToken)
            .put("token_type", "Bearer")
            .put("refresh_token", creds.refreshToken)
            .toString()
        return listOf(
            "[easycolab_gdrive]",
            "type = drive",
            "client_id = ${creds.clientId}",
            "client_secret = ${creds.clientSecret}",
            "scope = $driveScope",
            "token = $tokJson",
            ""
        ).joinToString("\n")
    }

    private suspend fun resolveOrStartKernelId(session: RuntimeSessionRecord, cleanExtra: Boolean = false): String =
        withContext(Dispatchers.IO) {
            val base = session.url.trimEnd('/')
            val kernelsUrl = "$base/api/kernels".toHttpUrl().newBuilder()
                .addQueryParameter("authuser", "0")
                .addQueryParameter("colab-runtime-proxy-token", session.token)
                .build()

            try {
                val listReq = Request.Builder()
                    .url(kernelsUrl)
                    .header("Accept", "application/json")
                    .header("X-Colab-Client-Agent", "colab-cli")
                    .header("X-Colab-Runtime-Proxy-Token", session.token)
                    .header("Authorization", "Bearer ${session.token}")
                    .build()
                httpClient.newCall(listReq).execute().use { resp ->
                    if (resp.isSuccessful) {
                        val arr = JSONArray(resp.body?.string().orEmpty())
                        val ids = (0 until arr.length()).mapNotNull { arr.optJSONObject(it)?.optStringOrNull("id") }
                        if (ids.isNotEmpty()) {
                            val primary = if (session.kernelId != null && ids.contains(session.kernelId)) session.kernelId else ids.first()
                            if (cleanExtra) {
                                for (extra in ids) {
                                    if (extra != primary) {
                                        val delUrl = "$base/api/kernels/$extra".toHttpUrl().newBuilder()
                                            .addQueryParameter("authuser", "0")
                                            .addQueryParameter("colab-runtime-proxy-token", session.token)
                                            .build()
                                        try {
                                            httpClient.newCall(
                                                Request.Builder().url(delUrl).delete()
                                                    .header("X-Colab-Client-Agent", "colab-cli")
                                                    .header("X-Colab-Runtime-Proxy-Token", session.token)
                                                    .header("Authorization", "Bearer ${session.token}")
                                                    .build()
                                            ).execute().close()
                                        } catch (_: Exception) {
                                        }
                                    }
                                }
                            }
                            return@withContext primary
                        }
                    }
                }
            } catch (_: Exception) {
            }

            val startReq = Request.Builder()
                .url(kernelsUrl)
                .post(JSONObject().put("name", "python3").toString().toRequestBody("application/json".toMediaType()))
                .header("Accept", "application/json")
                .header("X-Colab-Client-Agent", "colab-cli")
                .header("X-Colab-Runtime-Proxy-Token", session.token)
                .header("Authorization", "Bearer ${session.token}")
                .build()

            httpClient.newCall(startReq).execute().use { resp ->
                if (!resp.isSuccessful) throw ColabBridgeException("Failed to start Jupyter kernel (HTTP ${resp.code})", resp.code)
                val kid = JSONObject(resp.body?.string().orEmpty()).optString("id", "")
                if (kid.isEmpty()) throw ColabBridgeException("Empty kernel ID")
                kid
            }
        }

    private fun decodeJupyterBinaryMessage(bytes: ByteArray): JSONObject? {
        try {
            return JSONObject(String(bytes, Charsets.UTF_8))
        } catch (_: Exception) {
        }
        if (bytes.size < 12) return null
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        val nbufs = bb.int.toLong() and 0xFFFFFFFFL
        if (nbufs < 1 || bytes.size < 4 * (nbufs + 1)) return null
        val o0 = bb.int
        val o1 = if (nbufs >= 2) bb.int else bytes.size
        if (o0 < 0 || o1 < o0 || o1 > bytes.size) return null
        return try {
            JSONObject(String(bytes, o0, o1 - o0, Charsets.UTF_8))
        } catch (_: Exception) {
            null
        }
    }

    private suspend fun executeCodeOnKernel(
        session: RuntimeSessionRecord,
        code: String,
        timeoutMs: Long,
        onStdoutLine: ((String) -> Unit)? = null
    ): List<String> = withContext(Dispatchers.IO) {
        val kernelId = resolveOrStartKernelId(session, cleanExtra = true)
        val sessionId = session.sessionId ?: UUID.randomUUID().toString().lowercase(Locale.US)
        val updated = session.copy(kernelId = kernelId, sessionId = sessionId)
        saveSessionRecord(updated)

        val wsUrl = session.url.trimEnd('/').toHttpUrl().newBuilder()
            .addPathSegments("api/kernels/$kernelId/channels")
            .addQueryParameter("session_id", sessionId)
            .addQueryParameter("token", session.token)
            .addQueryParameter("colab-runtime-proxy-token", session.token)
            .addQueryParameter("authuser", "0")
            .build()

        val wsReq = Request.Builder()
            .url(wsUrl)
            .header("X-Colab-Client-Agent", "colab-cli")
            .header("X-Colab-Runtime-Proxy-Token", session.token)
            .header("Authorization", "Bearer ${session.token}")
            .build()

        val msgId = UUID.randomUUID().toString().lowercase(Locale.US)
        val execPayload = JSONObject()
            .put("channel", "shell")
            .put(
                "header", JSONObject()
                    .put("msg_id", msgId)
                    .put("username", "easycolab")
                    .put("session", sessionId)
                    .put("date", Instant.now().toString())
                    .put("msg_type", "execute_request")
                    .put("version", "5.3")
            )
            .put("parent_header", JSONObject())
            .put("metadata", JSONObject())
            .put(
                "content", JSONObject()
                    .put("code", code)
                    .put("silent", false)
                    .put("store_history", false)
                    .put("user_expressions", JSONObject())
                    .put("allow_stdin", false)
                    .put("stop_on_error", true)
            )
            .toString()

        val collected = mutableListOf<String>()
        val doneDeferred = CompletableDeferred<Unit>()

        val ws = httpClient.newWebSocket(wsReq, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send(execPayload)
            }

            private fun handleMsg(msg: JSONObject?) {
                if (msg == null || doneDeferred.isCompleted) return
                val parentId = msg.optJSONObject("parent_header")?.optString("msg_id", "")
                if (parentId != msgId) return

                val msgType = msg.optStringOrNull("msg_type")
                    ?: msg.optJSONObject("header")?.optString("msg_type", "").orEmpty()
                val content = msg.optJSONObject("content") ?: JSONObject()

                when (msgType) {
                    "stream" -> {
                        val text = content.optString("text", "")
                        for (rawLine in text.lines()) {
                            val trimmed = rawLine.trim()
                            if (trimmed.isEmpty()) continue
                            synchronized(collected) { collected.add(trimmed) }
                            onStdoutLine?.invoke(trimmed)
                            if (trimmed.startsWith("WAKE_URL:")) {
                                doneDeferred.complete(Unit)
                                return
                            }
                        }
                    }
                    "error" -> {
                        val eval = content.optString("evalue", "Kernel execution error")
                        val errLine = "WAKE_ERROR: $eval"
                        synchronized(collected) { collected.add(errLine) }
                        onStdoutLine?.invoke(errLine)
                        doneDeferred.complete(Unit)
                    }
                    "execute_reply" -> {
                        doneDeferred.complete(Unit)
                    }
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleMsg(try { JSONObject(text) } catch (_: Exception) { null })
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                handleMsg(decodeJupyterBinaryMessage(bytes.toByteArray()))
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (!doneDeferred.isCompleted) {
                    doneDeferred.completeExceptionally(t)
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (!doneDeferred.isCompleted) {
                    doneDeferred.complete(Unit)
                }
            }
        })

        try {
            withTimeoutOrNull(timeoutMs) {
                doneDeferred.await()
            }
        } finally {
            ws.close(1000, null)
        }
        synchronized(collected) { collected.toList() }
    }

    suspend fun wakeAndDeploy(
        gpuType: String,
        appPyBytes: ByteArray,
        onLog: (String) -> Unit
    ): String = withContext(Dispatchers.IO) {
        validAccessToken()
        val requestedAccel = (listOf("G4", "H100", "A100", "L4", "T4")
            .firstOrNull { it.equals(gpuType, ignoreCase = true) } ?: "G4").uppercase(Locale.US)

        onLog(L("正在检查账号下已有的 Colab 云端实例...", "Checking active Colab cloud instances on your account..."))
        val assignments = try { listAssignments() } catch (_: Exception) { emptyList() }
        var activeSession = loadSessionRecord()

        assignments.firstOrNull()?.let { first ->
            val ep = first.optString("endpoint", "")
            val acc = first.optString("accelerator", "")
            val proxy = first.optJSONObject("runtimeProxyInfo")
            val tok = proxy?.optString("token", "").orEmpty()
            val url = proxy?.optString("url", "").orEmpty()
            if (ep.isNotEmpty() && tok.isNotEmpty() && url.isNotEmpty()) {
                if (activeSession == null || activeSession?.endpoint == ep) {
                    activeSession = RuntimeSessionRecord(
                        name = "easycolab",
                        token = tok,
                        url = url,
                        endpoint = ep,
                        accelerator = acc.ifEmpty { requestedAccel },
                        kernelId = activeSession?.kernelId,
                        sessionId = activeSession?.sessionId
                    )
                    saveSessionRecord(activeSession)
                }
            }
        }

        activeSession?.let { cur ->
            if (!cur.accelerator.equals(requestedAccel, ignoreCase = true)) {
                onLog(L("当前实例型号为 [${cur.accelerator}]，正在释放并切换为 [$requestedAccel]...", "Releasing [${cur.accelerator}] and switching to [$requestedAccel]..."))
                unassignEndpoint(cur.endpoint)
                saveSessionRecord(null)
                activeSession = null
            } else {
                onLog(L("正在检查已有实例 (${cur.accelerator}) 连通性...", "Checking existing (${cur.accelerator}) runtime connectivity..."))
                if (!waitForRuntimeReady(cur, 15_000L)) {
                    onLog(L("旧实例已过期，正在自动清理并分配全新实例...", "Previous runtime expired; allocating a fresh VM..."))
                    unassignEndpoint(cur.endpoint)
                    saveSessionRecord(null)
                    activeSession = null
                }
            }
        }

        val targetSession = activeSession ?: allocateFreshSession(requestedAccel, onLog).also {
            waitForRuntimeReady(it, 50_000L)
        }

        onLog(L("正在上传最新多模型开源服务端代码 (app.py)...", "Uploading latest multi-model server engine (app.py)..."))
        var lastErr: Exception? = null
        for (attempt in 0 until 4) {
            try {
                uploadFileToColab(targetSession, appPyBytes, "content/app.py")
                lastErr = null
                break
            } catch (e: Exception) {
                lastErr = e
                if (attempt < 3) delay(2500)
            }
        }
        if (lastErr != null) throw lastErr

        buildRcloneDriveConf()?.let { conf ->
            try {
                uploadFileToColab(targetSession, conf.toByteArray(Charsets.UTF_8), "content/.easycolab_rclone.conf")
            } catch (_: Exception) {
            }
        }

        onLog(L("正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...", "Configuring Google Drive persistence, multi-model engine & public tunnel (~40–90s)..."))

        val bootCode = """
import os, sys, time, re, shutil, subprocess

rc_stage = "/content/.easycolab_rclone.conf"
if os.path.exists(rc_stage):
    rc_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(rc_dir, mode=0o700, exist_ok=True)
    rc_dst = os.path.join(rc_dir, "rclone.conf")
    shutil.move(rc_stage, rc_dst)
    os.chmod(rc_dst, 0o600)

if os.path.exists(os.path.expanduser("~/.config/rclone/rclone.conf")):
    if not shutil.which("rclone"):
        subprocess.run(
            ["bash", "-lc", "curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && unzip -q -o /tmp/rclone.zip -d /tmp && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone && rm -rf /tmp/rclone*"],
            check=False,
        )
    if shutil.which("rclone") and not os.path.ismount("/content/drive/MyDrive"):
        subprocess.run(["rclone", "mkdir", "easycolab_gdrive:EasyColab_Drive"], check=False)
        if os.path.exists("/dev/fuse") and (shutil.which("fusermount3") or shutil.which("fusermount")):
            os.makedirs("/content/drive/MyDrive", exist_ok=True)
            subprocess.Popen(
                [
                    "rclone", "mount", "easycolab_gdrive:EasyColab_Drive", "/content/drive/MyDrive",
                    "--vfs-cache-mode", "writes",
                    "--vfs-write-back", "3s",
                    "--dir-cache-time", "15s",
                    "--drive-chunk-size", "64M",
                    "--buffer-size", "32M",
                    "--allow-non-empty",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(10):
                if os.path.ismount("/content/drive/MyDrive"):
                    print("已通过 Google OAuth 自动挂载 Google Drive (/content/drive/MyDrive)！", flush=True)
                    break
                time.sleep(0.5)
            else:
                try:
                    if not os.listdir("/content/drive/MyDrive"):
                        os.rmdir("/content/drive/MyDrive")
                except Exception:
                    pass

if not os.path.exists("/content/DiffSynth-Studio"):
    print("正在克隆 DiffSynth-Studio...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/modelscope/DiffSynth-Studio.git", "/content/DiffSynth-Studio"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", "huggingface_hub", "hf_transfer", "qwen-vl-utils", "einops", "kernels>=0.17.2,<0.18.0", "av", "gradio", "modelscope", "imageio", "imageio-ffmpeg", "scipy", "soundfile", "librosa"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fla-core", "flash-linear-attention"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--only-binary=:all:", "causal-conv1d"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/DiffSynth-Studio"], check=True)
else:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        import kernels
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "kernels>=0.17.2,<0.18.0"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "fla-core", "flash-linear-attention"], check=False)

subprocess.run(["pkill", "-f", "python.*/content/app.py"], check=False)
subprocess.run(["pkill", "-f", "llama-server"], check=False)
time.sleep(1)

log_path = "/content/easycolab_server.log"
if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "w") as out_f:
    subprocess.Popen([sys.executable, "-u", "/content/app.py"], cwd="/content", stdout=out_f, stderr=subprocess.STDOUT)

url_re = re.compile(r"https://[a-zA-Z0-9.-]+\.gradio\.live")
found_url = None
for _ in range(90):
    time.sleep(2)
    if os.path.exists(log_path):
        txt = open(log_path, "r", errors="replace").read()
        m = url_re.search(txt)
        if m:
            found_url = m.group(0)
            break

if found_url:
    print("WAKE_URL:" + found_url, flush=True)
else:
    tail = open(log_path, "r", errors="replace").read()[-600:] if os.path.exists(log_path) else "no log"
    print("WAKE_ERROR: 启动超时，日志尾部: " + tail, flush=True)
        """.trimIndent()

        val lines = try {
            executeCodeOnKernel(targetSession, bootCode, timeoutMs = 300_000L) { line ->
                if (!line.startsWith("WAKE_URL:")) onLog(line.removePrefix("WAKE_LOG: "))
            }
        } catch (_: Exception) {
            val fresh = targetSession.copy(kernelId = null, sessionId = null)
            saveSessionRecord(fresh)
            executeCodeOnKernel(fresh, bootCode, timeoutMs = 300_000L) { line ->
                if (!line.startsWith("WAKE_URL:")) onLog(line.removePrefix("WAKE_LOG: "))
            }
        }

        lines.firstOrNull { it.startsWith("WAKE_URL:") }?.let {
            return@withContext it.removePrefix("WAKE_URL:").trim()
        }
        lines.firstOrNull { it.startsWith("WAKE_ERROR:") }?.let {
            throw ColabBridgeException(it)
        }
        throw ColabBridgeException(L("未能从云端提取 gradio.live 隧道地址，请重试", "Timed out waiting for gradio.live tunnel URL"))
    }

    suspend fun keepalive(): NativeAuthStatus = withContext(Dispatchers.IO) {
        if (loadCredentials() == null) {
            return@withContext NativeAuthStatus(ok = true, loggedIn = false, email = "")
        }
        try {
            listAssignments().firstOrNull()?.let { first ->
                val ep = first.optString("endpoint", "")
                val acc = first.optString("accelerator", "")
                val proxy = first.optJSONObject("runtimeProxyInfo")
                val tok = proxy?.optString("token", "").orEmpty()
                val url = proxy?.optString("url", "").orEmpty()
                if (ep.isNotEmpty() && tok.isNotEmpty() && url.isNotEmpty()) {
                    try {
                        colabTunnelRequest(
                            "/tun/m/$ep/keep-alive/",
                            method = "GET",
                            extraHeaders = mapOf("X-Colab-Tunnel" to "Google")
                        )
                    } catch (_: Exception) {
                    }
                    val prev = loadSessionRecord()
                    val updated = RuntimeSessionRecord("easycolab", tok, url, ep, acc, prev?.kernelId, prev?.sessionId)
                    saveSessionRecord(updated)
                    try {
                        executeCodeOnKernel(
                            updated,
                            "import pathlib; pathlib.Path('/tmp/.easycolab_client_hb').touch(); _easycolab_hb = 1",
                            timeoutMs = 12_000L
                        )
                    } catch (_: Exception) {
                    }
                }
            }
        } catch (_: Exception) {
        }
        fetchAuthStatus()
    }
}
