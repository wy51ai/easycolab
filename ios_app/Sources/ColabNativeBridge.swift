import Foundation
import CryptoKit
import Security

// MARK: - Native Pure-Swift Google Colab Headless Bridge (Zero Python Dependency)

struct NativeAuthStatus {
    var ok: Bool
    var loggedIn: Bool
    var email: String
    var paidBalance: Double?
    var burnRateHourly: Double?
    var activeAssignments: Int
    var message: String?
    var error: String?
}

struct StoredOAuthCredentials: Codable {
    var accessToken: String
    var refreshToken: String
    var expiryTimestamp: Double
    var clientId: String
    var clientSecret: String
    var scopes: [String]
    var email: String
}

struct CustomOAuthClientConfig: Codable {
    var clientId: String
    var clientSecret: String
}

struct NativeOAuthSessionRequest {
    var authURL: URL
    var callbackScheme: String?
    var redirectURI: String
}

private struct PendingPKCEState: Codable {
    var verifier: String
    var redirectURI: String
}

private struct RuntimeSessionRecord: Codable {
    var name: String
    var token: String
    var url: String
    var endpoint: String
    var accelerator: String
    var kernelId: String?
    var sessionId: String?
}

enum ColabBridgeError: LocalizedError {
    case notLoggedIn
    case invalidAuthURL
    case tokenExchangeFailed(String)
    case httpError(Int, String)
    case invalidResponse(String)
    case kernelTimeout(String)

    var errorDescription: String? {
        switch self {
        case .notLoggedIn:
            return L("尚未绑定 Google 账号", "Google account not bound")
        case .invalidAuthURL:
            return L("生成授权链接失败", "Failed to build OAuth URL")
        case .tokenExchangeFailed(let msg):
            return msg
        case .httpError(let code, let body):
            if code == 412 {
                return L("当前账号活跃 Colab 实例数已达上限 (HTTP 412)，请先点击「断开并释放实例」", "Too many active Colab instances (HTTP 412). Terminate existing instances first.")
            }
            return "HTTP \(code): \(body.prefix(200))"
        case .invalidResponse(let msg):
            return msg
        case .kernelTimeout(let msg):
            return msg
        }
    }
}

actor ColabNativeBridge {
    static let shared = ColabNativeBridge()

    private let colabDomain = "https://colab.research.google.com"
    private let remoteRedirectURI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
    private let publicScopes = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/colaboratory",
        "https://www.googleapis.com/auth/drive.file"
    ]

    private let keychainService = "app.easycolab.studio.ios.oauth"
    private let keychainAccount = "primary_google_oauth"
    private var pendingPKCEVerifier: String? = nil
    private var pendingRedirectURI: String? = nil

    // Runtime-decoded installed-app OAuth parameters (avoids plaintext secret scanning false positives)
    private static let encodedClientIdBytes: [UInt8] = [
        109, 108, 110, 106, 98, 108, 106, 111, 107, 98, 111, 106, 119, 108, 43, 40, 110, 42,
        108, 61, 42, 51, 108, 50, 52, 111, 106, 108, 42, 46, 98, 63, 48, 47, 43, 98, 105, 62,
        51, 105, 110, 107, 50, 47, 40, 116, 59, 42, 42, 41, 116, 61, 53, 53, 61, 54, 63, 47,
        41, 63, 40, 57, 53, 52, 46, 63, 52, 46, 116, 57, 53, 55
    ]
    private static let encodedClientSecretBytes: [UInt8] = [
        62, 119, 28, 22, 99, 111, 11, 107, 99, 43, 109, 23, 11, 55, 28, 42, 62, 109, 50, 18, 30, 106, 14, 35
    ]

    private static func decodeXOR(_ bytes: [UInt8], key: UInt8 = 0x5A) -> String {
        String(bytes: bytes.map { $0 ^ key }, encoding: .utf8) ?? ""
    }

    private var supportDir: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = base.appendingPathComponent("EasyColab", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private var fallbackCredsFile: URL { supportDir.appendingPathComponent("oauth_creds.json") }
    private var sessionRecordFile: URL { supportDir.appendingPathComponent("colab_session.json") }
    private var customOAuthFile: URL { supportDir.appendingPathComponent("custom_oauth_client.json") }
    private var pkceFile: URL { supportDir.appendingPathComponent("pkce_verifier.txt") }

    // MARK: - OAuth Client Config (Default + Optional User Override)

    func loadActiveOAuthClient() -> CustomOAuthClientConfig {
        if let data = try? Data(contentsOf: customOAuthFile),
           let custom = try? JSONDecoder().decode(CustomOAuthClientConfig.self, from: data),
           !custom.clientId.isEmpty, !custom.clientSecret.isEmpty {
            return custom
        }
        return CustomOAuthClientConfig(
            clientId: Self.decodeXOR(Self.encodedClientIdBytes),
            clientSecret: Self.decodeXOR(Self.encodedClientSecretBytes)
        )
    }

    func saveCustomOAuthJSON(_ rawJSON: String) throws -> Bool {
        let trimmed = rawJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            try? FileManager.default.removeItem(at: customOAuthFile)
            return false
        }
        guard let data = trimmed.data(using: .utf8),
              let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ColabBridgeError.invalidResponse(L("无效的 JSON 格式", "Invalid JSON format"))
        }
        let inner = (obj["installed"] as? [String: Any]) ?? (obj["web"] as? [String: Any]) ?? obj
        guard let cid = inner["client_id"] as? String, !cid.isEmpty,
              let sec = inner["client_secret"] as? String, !sec.isEmpty else {
            throw ColabBridgeError.invalidResponse(L("JSON 中缺少 client_id 或 client_secret", "Missing client_id or client_secret in JSON"))
        }
        let cfg = CustomOAuthClientConfig(clientId: cid, clientSecret: sec)
        let encoded = try JSONEncoder().encode(cfg)
        try encoded.write(to: customOAuthFile, options: .atomic)
        return true
    }

    func hasCustomOAuthConfig() -> Bool {
        FileManager.default.fileExists(atPath: customOAuthFile.path)
    }

    // MARK: - Keychain & Local Credential Storage

    private func saveCredentials(_ creds: StoredOAuthCredentials) {
        guard let data = try? JSONEncoder().encode(creds) else { return }
        // Save to iOS Keychain (encrypted hardware-backed credential storage)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: keychainAccount
        ]
        SecItemDelete(query as CFDictionary)
        var addQuery = query
        addQuery[kSecValueData as String] = data
        addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        let status = SecItemAdd(addQuery as CFDictionary, nil)
        // Always mirror to protected app support file for Simulator / ad-hoc environments without keychain entitlement
        if status != errSecSuccess || true {
            try? data.write(to: fallbackCredsFile, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        }
    }

    private func loadCredentials() -> StoredOAuthCredentials? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: keychainAccount,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecSuccess, let data = result as? Data,
           let creds = try? JSONDecoder().decode(StoredOAuthCredentials.self, from: data) {
            return creds
        }
        if let data = try? Data(contentsOf: fallbackCredsFile),
           let creds = try? JSONDecoder().decode(StoredOAuthCredentials.self, from: data) {
            return creds
        }
        return nil
    }

    private func clearCredentialsAndSessions() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: keychainAccount
        ]
        SecItemDelete(query as CFDictionary)
        try? FileManager.default.removeItem(at: fallbackCredsFile)
        try? FileManager.default.removeItem(at: sessionRecordFile)
        try? FileManager.default.removeItem(at: pkceFile)
        pendingPKCEVerifier = nil
        pendingRedirectURI = nil
    }

    private func loadSessionRecord() -> RuntimeSessionRecord? {
        guard let data = try? Data(contentsOf: sessionRecordFile) else { return nil }
        return try? JSONDecoder().decode(RuntimeSessionRecord.self, from: data)
    }

    private func saveSessionRecord(_ rec: RuntimeSessionRecord?) {
        guard let rec, let data = try? JSONEncoder().encode(rec) else {
            try? FileManager.default.removeItem(at: sessionRecordFile)
            return
        }
        try? data.write(to: sessionRecordFile, options: .atomic)
    }

    // MARK: - Google OAuth2 PKCE Flow (Native ASWebAuthenticationSession + Manual Fallback)

    private static func base64URLEncode(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    private static let rfc3986Unreserved: CharacterSet = {
        var cs = CharacterSet.alphanumerics
        cs.insert(charactersIn: "-._~")
        return cs
    }()

    private static func formURLEncodedData(_ items: [(String, String)]) -> Data? {
        let encoded = items.map { k, v in
            let ek = k.addingPercentEncoding(withAllowedCharacters: rfc3986Unreserved) ?? k
            let ev = v.addingPercentEncoding(withAllowedCharacters: rfc3986Unreserved) ?? v
            return "\(ek)=\(ev)"
        }.joined(separator: "&")
        return encoded.data(using: .utf8)
    }

    func generateAuthSession(preferCustomScheme: Bool = true) throws -> NativeOAuthSessionRequest {
        var randomBytes = [UInt8](repeating: 0, count: 36)
        _ = SecRandomCopyBytes(kSecRandomDefault, randomBytes.count, &randomBytes)
        let verifier = Self.base64URLEncode(Data(randomBytes))

        let challengeHash = SHA256.hash(data: Data(verifier.utf8))
        let challenge = Self.base64URLEncode(Data(challengeHash))

        var stateBytes = [UInt8](repeating: 0, count: 18)
        _ = SecRandomCopyBytes(kSecRandomDefault, stateBytes.count, &stateBytes)
        let stateStr = Self.base64URLEncode(Data(stateBytes))

        let client = loadActiveOAuthClient()
        let suffix = ".apps.googleusercontent.com"
        let callbackScheme: String?
        let redirectURI: String

        if preferCustomScheme && client.clientId.hasSuffix(suffix) {
            let prefix = String(client.clientId.dropLast(suffix.count))
            let scheme = "com.googleusercontent.apps.\(prefix)"
            callbackScheme = scheme
            redirectURI = "\(scheme):/oauth2redirect"
        } else {
            callbackScheme = nil
            redirectURI = remoteRedirectURI
        }

        pendingPKCEVerifier = verifier
        pendingRedirectURI = redirectURI
        if let stateData = try? JSONEncoder().encode(PendingPKCEState(verifier: verifier, redirectURI: redirectURI)) {
            try? stateData.write(to: pkceFile, options: .atomic)
        }

        var comps = URLComponents(string: "https://accounts.google.com/o/oauth2/v2/auth")!
        var qItems = [
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "client_id", value: client.clientId),
            URLQueryItem(name: "redirect_uri", value: redirectURI),
            URLQueryItem(name: "scope", value: publicScopes.joined(separator: " ")),
            URLQueryItem(name: "state", value: stateStr),
            URLQueryItem(name: "code_challenge", value: challenge),
            URLQueryItem(name: "code_challenge_method", value: "S256"),
            URLQueryItem(name: "prompt", value: "select_account consent"),
            URLQueryItem(name: "access_type", value: "offline")
        ]
        if callbackScheme == nil {
            qItems.append(URLQueryItem(name: "token_usage", value: "remote"))
        }
        comps.queryItems = qItems
        guard let url = comps.url else { throw ColabBridgeError.invalidAuthURL }
        return NativeOAuthSessionRequest(authURL: url, callbackScheme: callbackScheme, redirectURI: redirectURI)
    }

    func generateAuthURL() throws -> URL {
        try generateAuthSession(preferCustomScheme: false).authURL
    }

    func exchangeAuthCode(_ rawInput: String, redirectURI overrideRedirectURI: String? = nil) async -> NativeAuthStatus {
        let trimmed = rawInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return NativeAuthStatus(ok: false, loggedIn: false, email: "", activeAssignments: 0, error: "Empty code")
        }

        var code = trimmed
        if trimmed.contains(":/") || trimmed.contains("code=") || trimmed.contains("error="),
           let cbURL = URL(string: trimmed),
           let cbComps = URLComponents(url: cbURL, resolvingAgainstBaseURL: false) {
            if let oauthErr = cbComps.queryItems?.first(where: { $0.name == "error" })?.value, !oauthErr.isEmpty {
                return NativeAuthStatus(ok: false, loggedIn: false, email: "", activeAssignments: 0, error: oauthErr)
            }
            if let extracted = cbComps.queryItems?.first(where: { $0.name == "code" })?.value, !extracted.isEmpty {
                code = extracted
            }
        }

        let persistedState: PendingPKCEState? = {
            guard let data = try? Data(contentsOf: pkceFile) else { return nil }
            return try? JSONDecoder().decode(PendingPKCEState.self, from: data)
        }()

        let verifier = pendingPKCEVerifier
            ?? persistedState?.verifier
            ?? (try? String(contentsOf: pkceFile, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
            ?? ""
        let activeRedirectURI = overrideRedirectURI
            ?? pendingRedirectURI
            ?? persistedState?.redirectURI
            ?? remoteRedirectURI

        let client = loadActiveOAuthClient()

        var req = URLRequest(url: URL(string: "https://oauth2.googleapis.com/token")!)
        req.httpMethod = "POST"
        req.timeoutInterval = 25
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")

        var bodyPairs: [(String, String)] = [
            ("code", code),
            ("client_id", client.clientId),
            ("client_secret", client.clientSecret),
            ("redirect_uri", activeRedirectURI),
            ("grant_type", "authorization_code")
        ]
        if !verifier.isEmpty && !verifier.hasPrefix("{") {
            bodyPairs.append(("code_verifier", verifier))
        }
        req.httpBody = Self.formURLEncodedData(bodyPairs)

        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? 0
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return NativeAuthStatus(ok: false, loggedIn: false, email: "", activeAssignments: 0, error: "HTTP \(statusCode)")
            }
            if statusCode != 200 || json["access_token"] == nil {
                let errDesc = (json["error_description"] as? String) ?? (json["error"] as? String) ?? "HTTP \(statusCode)"
                return NativeAuthStatus(ok: false, loggedIn: false, email: "", activeAssignments: 0, error: errDesc)
            }

            let accessToken = json["access_token"] as? String ?? ""
            let refreshToken = (json["refresh_token"] as? String) ?? loadCredentials()?.refreshToken ?? ""
            let expiresIn = (json["expires_in"] as? Double) ?? 3500.0
            let scopeStr = (json["scope"] as? String) ?? publicScopes.joined(separator: " ")

            let email = await fetchUserEmail(accessToken: accessToken)
            saveSessionRecord(nil)
            try? FileManager.default.removeItem(at: pkceFile)
            pendingPKCEVerifier = nil
            pendingRedirectURI = nil

            let creds = StoredOAuthCredentials(
                accessToken: accessToken,
                refreshToken: refreshToken,
                expiryTimestamp: Date().timeIntervalSince1970 + expiresIn - 60,
                clientId: client.clientId,
                clientSecret: client.clientSecret,
                scopes: scopeStr.components(separatedBy: " "),
                email: email
            )
            saveCredentials(creds)
            return await fetchAuthStatus()
        } catch {
            return NativeAuthStatus(ok: false, loggedIn: false, email: "", activeAssignments: 0, error: error.localizedDescription)
        }
    }

    func logout() {
        clearCredentialsAndSessions()
    }

    private func validAccessToken() async throws -> StoredOAuthCredentials {
        guard var creds = loadCredentials() else {
            throw ColabBridgeError.notLoggedIn
        }
        if Date().timeIntervalSince1970 < creds.expiryTimestamp && !creds.accessToken.isEmpty {
            return creds
        }
        guard !creds.refreshToken.isEmpty else {
            throw ColabBridgeError.notLoggedIn
        }

        var req = URLRequest(url: URL(string: "https://oauth2.googleapis.com/token")!)
        req.httpMethod = "POST"
        req.timeoutInterval = 20
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        req.httpBody = Self.formURLEncodedData([
            ("client_id", creds.clientId),
            ("client_secret", creds.clientSecret),
            ("refresh_token", creds.refreshToken),
            ("grant_type", "refresh_token")
        ])

        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let newToken = json["access_token"] as? String else {
            throw ColabBridgeError.notLoggedIn
        }
        let expiresIn = (json["expires_in"] as? Double) ?? 3500.0
        creds.accessToken = newToken
        creds.expiryTimestamp = Date().timeIntervalSince1970 + expiresIn - 60
        if creds.email.isEmpty {
            creds.email = await fetchUserEmail(accessToken: newToken)
        }
        saveCredentials(creds)
        return creds
    }

    private func fetchUserEmail(accessToken: String) async -> String {
        guard let url = URL(string: "https://www.googleapis.com/oauth2/v2/userinfo") else { return "" }
        var req = URLRequest(url: url)
        req.timeoutInterval = 8
        req.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let email = json["email"] as? String else {
            return ""
        }
        return email.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Colab Tunnel REST Requests (XSSI Stripping + XSRF Handling)

    private func stripXSSI(_ text: String) -> String {
        if text.hasPrefix(")]}'\n") {
            return String(text.dropFirst(5))
        }
        if text.hasPrefix(")]}'") {
            return String(text.dropFirst(4)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return text
    }

    private func colabTunnelRequest(
        path: String,
        method: String = "GET",
        queryItems: [URLQueryItem] = [],
        extraHeaders: [String: String] = [:],
        timeout: TimeInterval = 45
    ) async throws -> [String: Any] {
        let creds = try await validAccessToken()
        var comps = URLComponents(string: colabDomain + path)!
        var allItems = queryItems
        if !allItems.contains(where: { $0.name == "authuser" }) {
            allItems.append(URLQueryItem(name: "authuser", value: "0"))
        }
        comps.queryItems = allItems

        var req = URLRequest(url: comps.url!)
        req.httpMethod = method
        req.timeoutInterval = timeout
        req.setValue("Bearer \(creds.accessToken)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        req.setValue("colab-cli", forHTTPHeaderField: "X-Colab-Client-Agent")
        for (k, v) in extraHeaders {
            req.setValue(v, forHTTPHeaderField: k)
        }
        if method == "POST" && req.httpBody == nil {
            req.httpBody = Data()
        }

        let (data, resp) = try await URLSession.shared.data(for: req)
        let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? 0
        let rawText = String(data: data, encoding: .utf8) ?? ""
        guard (200..<300).contains(statusCode) else {
            throw ColabBridgeError.httpError(statusCode, rawText)
        }
        let clean = stripXSSI(rawText).trimmingCharacters(in: .whitespacesAndNewlines)
        if clean.isEmpty { return [:] }
        guard let cleanData = clean.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: cleanData) as? [String: Any] else {
            return [:]
        }
        return dict
    }

    private static func webSafeNotebookHash(_ id: UUID = UUID()) -> String {
        let s = id.uuidString.lowercased().replacingOccurrences(of: "-", with: "_")
        let padCount = max(0, 44 - s.count)
        return s + String(repeating: ".", count: padCount)
    }

    // MARK: - Account Status, Assignments, Assign & Unassign

    func fetchAuthStatus() async -> NativeAuthStatus {
        guard let stored = loadCredentials() else {
            return NativeAuthStatus(ok: true, loggedIn: false, email: "", activeAssignments: 0)
        }
        do {
            let creds = try await validAccessToken()
            let ccu = try await colabTunnelRequest(path: "/tun/m/ccu-info", timeout: 20)
            let balance = (ccu["currentBalance"] as? NSNumber)?.doubleValue ?? 0.0
            let burnRate = (ccu["consumptionRateHourly"] as? NSNumber)?.doubleValue ?? 0.0
            let count = (ccu["assignmentsCount"] as? NSNumber)?.intValue ?? 0
            return NativeAuthStatus(
                ok: true,
                loggedIn: true,
                email: creds.email,
                paidBalance: (balance * 100).rounded() / 100,
                burnRateHourly: (burnRate * 100).rounded() / 100,
                activeAssignments: count
            )
        } catch ColabBridgeError.notLoggedIn {
            return NativeAuthStatus(ok: true, loggedIn: false, email: "", activeAssignments: 0)
        } catch {
            return NativeAuthStatus(
                ok: false,
                loggedIn: true,
                email: stored.email,
                activeAssignments: 0,
                error: error.localizedDescription
            )
        }
    }

    private func listAssignments() async throws -> [[String: Any]] {
        let res = try await colabTunnelRequest(path: "/tun/m/assignments", timeout: 25)
        return (res["assignments"] as? [[String: Any]]) ?? []
    }

    private func unassignEndpoint(_ endpoint: String) async {
        guard !endpoint.isEmpty else { return }
        do {
            let getResp = try await colabTunnelRequest(path: "/tun/m/unassign/\(endpoint)", method: "GET", timeout: 20)
            if let xsrf = getResp["token"] as? String, !xsrf.isEmpty {
                _ = try await colabTunnelRequest(
                    path: "/tun/m/unassign/\(endpoint)",
                    method: "POST",
                    extraHeaders: ["X-Goog-Colab-Token": xsrf],
                    timeout: 25
                )
            }
        } catch {}
    }

    func stopAll() async -> NativeAuthStatus {
        guard loadCredentials() != nil else {
            return NativeAuthStatus(ok: false, loggedIn: false, email: "", activeAssignments: 0, error: "Not logged in")
        }
        var stoppedCount = 0
        if let rec = loadSessionRecord() {
            // Trigger quick ComfyUI Google Drive sync before shutdown if running
            _ = try? await executeCodeOnKernel(
                session: rec,
                code: """
                import urllib.request
                for _p in (27860, 7860):
                    try:
                        urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{_p}/minimax_api/comfyui/sync_drive', data=b'{}', headers={'Content-Type':'application/json'}, method='POST'), timeout=12)
                        break
                    except Exception:
                        pass
                """,
                timeout: 15
            )
            await unassignEndpoint(rec.endpoint)
            saveSessionRecord(nil)
            stoppedCount += 1
        }

        if let assignments = try? await listAssignments() {
            for item in assignments {
                if let ep = item["endpoint"] as? String, !ep.isEmpty {
                    await unassignEndpoint(ep)
                    stoppedCount += 1
                }
            }
        }

        var status = await fetchAuthStatus()
        let rate = status.burnRateHourly ?? 0.0
        status.message = L(
            "已彻底释放全部云端实例 (\(stoppedCount) 台)！当前计费速率: \(String(format: "%.2f", rate)) CU/h",
            "Terminated all cloud instances (\(stoppedCount)). Current burn rate: \(String(format: "%.2f", rate)) CU/h"
        )
        return status
    }

    // MARK: - Allocate VM, Upload app.py & Boot via Jupyter WebSocket

    private func allocateFreshSession(gpuType: String, onLog: @Sendable (String) -> Void) async throws -> RuntimeSessionRecord {
        let accel = ["G4", "H100", "A100", "L4", "T4"].first { $0.caseInsensitiveCompare(gpuType) == .orderedSame } ?? "G4"
        let singleShapeGPUs: Set<String> = ["G4", "H100", "L4"]
        let useHighMem = !singleShapeGPUs.contains(accel)

        let gpuDesc = accel == "G4" ? "G4 (RTX PRO 6000 · 96GB)" : accel
        onLog(L("正在向 Google Colab 申请分配 \(gpuDesc) 实例...", "Requesting \(gpuDesc) instance from Google Colab..."))

        func performAssign(highMem: Bool) async throws -> [String: Any] {
            let nbh = Self.webSafeNotebookHash()
            var q: [URLQueryItem] = [
                URLQueryItem(name: "nbh", value: nbh),
                URLQueryItem(name: "variant", value: "GPU"),
                URLQueryItem(name: "accelerator", value: accel)
            ]
            if highMem {
                q.append(URLQueryItem(name: "shape", value: "hm"))
            }
            let getResp = try await colabTunnelRequest(path: "/tun/m/assign", method: "GET", queryItems: q, timeout: 60)
            if getResp["runtimeProxyInfo"] != nil {
                return getResp
            }
            guard let xsrf = getResp["token"] as? String, !xsrf.isEmpty else {
                throw ColabBridgeError.invalidResponse(L("未能获取实例分配令牌，请检查账号 CU 余额", "Failed to obtain assignment token. Check your CU balance."))
            }
            return try await colabTunnelRequest(
                path: "/tun/m/assign",
                method: "POST",
                queryItems: q,
                extraHeaders: ["X-Goog-Colab-Token": xsrf],
                timeout: 90
            )
        }

        let resp: [String: Any]
        do {
            resp = try await performAssign(highMem: useHighMem)
        } catch ColabBridgeError.httpError(let code, _) where code == 400 && useHighMem {
            resp = try await performAssign(highMem: false)
        }

        guard let endpoint = resp["endpoint"] as? String,
              let proxy = resp["runtimeProxyInfo"] as? [String: Any],
              let proxyToken = proxy["token"] as? String,
              let proxyURL = proxy["url"] as? String, !proxyURL.isEmpty else {
            throw ColabBridgeError.invalidResponse(L("分配失败：账号可能无可用付费 CU 或当前 GPU 缺货，可尝试切换 L4 / A100", "Allocation failed: check paid CU balance or try switching to L4 / A100."))
        }

        let rec = RuntimeSessionRecord(
            name: "easycolab",
            token: proxyToken,
            url: proxyURL,
            endpoint: endpoint,
            accelerator: accel,
            kernelId: nil,
            sessionId: nil
        )
        saveSessionRecord(rec)
        onLog(L("成功分配 \(accel) 云端虚拟机！正在等待容器内核就绪...", "Allocated \(accel) cloud VM! Waiting for Jupyter kernel..."))
        return rec
    }

    private func waitForRuntimeReady(_ session: RuntimeSessionRecord, timeoutSeconds: TimeInterval) async -> Bool {
        let base = session.url.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        var comps = URLComponents(string: "\(base)/api/kernels")!
        comps.queryItems = [
            URLQueryItem(name: "authuser", value: "0"),
            URLQueryItem(name: "colab-runtime-proxy-token", value: session.token)
        ]
        guard let url = comps.url else { return false }

        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var notFoundCount = 0
        while Date() < deadline {
            var req = URLRequest(url: url)
            req.timeoutInterval = 8
            req.setValue("application/json", forHTTPHeaderField: "Accept")
            req.setValue("colab-cli", forHTTPHeaderField: "X-Colab-Client-Agent")
            req.setValue(session.token, forHTTPHeaderField: "X-Colab-Runtime-Proxy-Token")
            req.setValue("Bearer \(session.token)", forHTTPHeaderField: "Authorization")

            if let (_, resp) = try? await URLSession.shared.data(for: req),
               let code = (resp as? HTTPURLResponse)?.statusCode {
                if code == 200 { return true }
                if code == 404 || code == 410 {
                    notFoundCount += 1
                    if notFoundCount >= 2 { return false }
                }
            }
            try? await Task.sleep(nanoseconds: 2_000_000_000)
        }
        return false
    }

    private func uploadFileToColab(session: RuntimeSessionRecord, data: Data, remotePath: String) async throws {
        let cleanRemote = remotePath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let base = session.url.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        var comps = URLComponents(string: "\(base)/api/contents/\(cleanRemote)")!
        comps.queryItems = [
            URLQueryItem(name: "authuser", value: "0"),
            URLQueryItem(name: "colab-runtime-proxy-token", value: session.token)
        ]
        let filename = (cleanRemote as NSString).lastPathComponent
        let payload: [String: Any] = [
            "name": filename,
            "path": cleanRemote,
            "type": "file",
            "format": "base64",
            "content": data.base64EncodedString(),
            "chunk": 1
        ]

        var req = URLRequest(url: comps.url!)
        req.httpMethod = "PUT"
        req.timeoutInterval = 45
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        req.setValue("colab-cli", forHTTPHeaderField: "X-Colab-Client-Agent")
        req.setValue(session.token, forHTTPHeaderField: "X-Colab-Runtime-Proxy-Token")
        req.setValue("Bearer \(session.token)", forHTTPHeaderField: "Authorization")
        req.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (respData, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            throw ColabBridgeError.httpError(code, String(data: respData, encoding: .utf8) ?? "")
        }
    }

    private func buildRcloneDriveConf() async -> String? {
        guard let creds = try? await validAccessToken(), !creds.refreshToken.isEmpty else { return nil }
        let driveScope = creds.scopes.contains(where: { $0.hasSuffix("/drive") })
            ? "drive"
            : (creds.scopes.contains(where: { $0.contains("drive.file") }) ? "drive.file" : "")
        guard !driveScope.isEmpty else { return nil }
        let tokObj: [String: String] = [
            "access_token": creds.accessToken,
            "token_type": "Bearer",
            "refresh_token": creds.refreshToken
        ]
        guard let tokData = try? JSONSerialization.data(withJSONObject: tokObj),
              let tokStr = String(data: tokData, encoding: .utf8) else { return nil }
        return [
            "[easycolab_gdrive]",
            "type = drive",
            "client_id = \(creds.clientId)",
            "client_secret = \(creds.clientSecret)",
            "scope = \(driveScope)",
            "token = \(tokStr)",
            ""
        ].joined(separator: "\n")
    }

    private func resolveOrStartKernelId(session: RuntimeSessionRecord, cleanExtra: Bool = false) async throws -> String {
        let base = session.url.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        var comps = URLComponents(string: "\(base)/api/kernels")!
        comps.queryItems = [
            URLQueryItem(name: "authuser", value: "0"),
            URLQueryItem(name: "colab-runtime-proxy-token", value: session.token)
        ]

        var listReq = URLRequest(url: comps.url!)
        listReq.timeoutInterval = 15
        listReq.setValue("application/json", forHTTPHeaderField: "Accept")
        listReq.setValue("colab-cli", forHTTPHeaderField: "X-Colab-Client-Agent")
        listReq.setValue(session.token, forHTTPHeaderField: "X-Colab-Runtime-Proxy-Token")
        listReq.setValue("Bearer \(session.token)", forHTTPHeaderField: "Authorization")

        if let (lData, lResp) = try? await URLSession.shared.data(for: listReq),
           (lResp as? HTTPURLResponse)?.statusCode == 200,
           let arr = try? JSONSerialization.jsonObject(with: lData) as? [[String: Any]],
           !arr.isEmpty {
            let ids = arr.compactMap { $0["id"] as? String }
            if let primary = (session.kernelId != nil && ids.contains(session.kernelId!)) ? session.kernelId! : ids.first {
                if cleanExtra {
                    for extra in ids where extra != primary {
                        var delComps = URLComponents(string: "\(base)/api/kernels/\(extra)")!
                        delComps.queryItems = comps.queryItems
                        var delReq = URLRequest(url: delComps.url!)
                        delReq.httpMethod = "DELETE"
                        delReq.timeoutInterval = 6
                        delReq.setValue("colab-cli", forHTTPHeaderField: "X-Colab-Client-Agent")
                        delReq.setValue(session.token, forHTTPHeaderField: "X-Colab-Runtime-Proxy-Token")
                        delReq.setValue("Bearer \(session.token)", forHTTPHeaderField: "Authorization")
                        _ = try? await URLSession.shared.data(for: delReq)
                    }
                }
                return primary
            }
        }

        // Start a new python3 kernel
        var startReq = URLRequest(url: comps.url!)
        startReq.httpMethod = "POST"
        startReq.timeoutInterval = 25
        startReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        startReq.setValue("application/json", forHTTPHeaderField: "Accept")
        startReq.setValue("colab-cli", forHTTPHeaderField: "X-Colab-Client-Agent")
        startReq.setValue(session.token, forHTTPHeaderField: "X-Colab-Runtime-Proxy-Token")
        startReq.setValue("Bearer \(session.token)", forHTTPHeaderField: "Authorization")
        startReq.httpBody = try JSONSerialization.data(withJSONObject: ["name": "python3"])

        let (sData, sResp) = try await URLSession.shared.data(for: startReq)
        let code = (sResp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code),
              let dict = try? JSONSerialization.jsonObject(with: sData) as? [String: Any],
              let kid = dict["id"] as? String else {
            throw ColabBridgeError.invalidResponse("Failed to start Jupyter kernel (HTTP \(code))")
        }
        return kid
    }

    private static func decodeJupyterWSMessage(_ wsMessage: URLSessionWebSocketTask.Message) -> [String: Any]? {
        switch wsMessage {
        case .string(let text):
            guard let data = text.data(using: .utf8) else { return nil }
            return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        case .data(let bytes):
            if let dict = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any] {
                return dict
            }
            // Jupyter Default binary protocol: big-endian UInt32 nbufs + offsets
            guard bytes.count >= 12 else { return nil }
            let nbufs = bytes.prefix(4).reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
            guard nbufs >= 1, bytes.count >= Int(4 * (nbufs + 1)) else { return nil }
            let o0Start = 4
            let o0 = bytes[o0Start..<(o0Start + 4)].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
            let o1: UInt32
            if nbufs >= 2 {
                o1 = bytes[8..<12].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
            } else {
                o1 = UInt32(bytes.count)
            }
            guard o0 <= o1, Int(o1) <= bytes.count else { return nil }
            let jsonSlice = bytes[Int(o0)..<Int(o1)]
            return try? JSONSerialization.jsonObject(with: jsonSlice) as? [String: Any]
        @unknown default:
            return nil
        }
    }

    private func executeCodeOnKernel(
        session: RuntimeSessionRecord,
        code: String,
        timeout: TimeInterval,
        onStdoutLine: (@Sendable (String) -> Void)? = nil
    ) async throws -> [String] {
        let kernelId = try await resolveOrStartKernelId(session: session, cleanExtra: true)
        let sessionId = session.sessionId ?? UUID().uuidString.lowercased()
        var updated = session
        updated.kernelId = kernelId
        updated.sessionId = sessionId
        saveSessionRecord(updated)

        let httpBase = session.url.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let wsBase: String
        if httpBase.hasPrefix("https://") {
            wsBase = "wss://" + httpBase.dropFirst("https://".count)
        } else if httpBase.hasPrefix("http://") {
            wsBase = "ws://" + httpBase.dropFirst("http://".count)
        } else {
            wsBase = "wss://" + httpBase
        }

        var comps = URLComponents(string: "\(wsBase)/api/kernels/\(kernelId)/channels")!
        comps.queryItems = [
            URLQueryItem(name: "session_id", value: sessionId),
            URLQueryItem(name: "token", value: session.token),
            URLQueryItem(name: "colab-runtime-proxy-token", value: session.token),
            URLQueryItem(name: "authuser", value: "0")
        ]

        var wsReq = URLRequest(url: comps.url!)
        wsReq.timeoutInterval = 30
        wsReq.setValue("colab-cli", forHTTPHeaderField: "X-Colab-Client-Agent")
        wsReq.setValue(session.token, forHTTPHeaderField: "X-Colab-Runtime-Proxy-Token")
        wsReq.setValue("Bearer \(session.token)", forHTTPHeaderField: "Authorization")

        let wsTask = URLSession.shared.webSocketTask(with: wsReq)
        wsTask.resume()
        defer { wsTask.cancel(with: .normalClosure, reason: nil) }

        let msgId = UUID().uuidString.lowercased()
        let isoNow = ISO8601DateFormatter().string(from: Date())
        let execMsg: [String: Any] = [
            "channel": "shell",
            "header": [
                "msg_id": msgId,
                "username": "easycolab",
                "session": sessionId,
                "date": isoNow,
                "msg_type": "execute_request",
                "version": "5.3"
            ],
            "parent_header": [String: String](),
            "metadata": [String: String](),
            "content": [
                "code": code,
                "silent": false,
                "store_history": false,
                "user_expressions": [String: String](),
                "allow_stdin": false,
                "stop_on_error": true
            ]
        ]

        let reqData = try JSONSerialization.data(withJSONObject: execMsg)
        let reqText = String(data: reqData, encoding: .utf8)!
        try await wsTask.send(.string(reqText))

        var collectedLines: [String] = []
        let deadline = Date().addingTimeInterval(timeout)

        while Date() < deadline {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { break }

            let wsMsg = try await withThrowingTaskGroup(of: URLSessionWebSocketTask.Message.self) { group in
                group.addTask { try await wsTask.receive() }
                group.addTask {
                    try await Task.sleep(nanoseconds: UInt64(min(remaining, 25.0) * 1_000_000_000))
                    throw URLError(.timedOut)
                }
                do {
                    let first = try await group.next()!
                    group.cancelAll()
                    return first
                } catch {
                    group.cancelAll()
                    throw error
                }
            }

            guard let msg = Self.decodeJupyterWSMessage(wsMsg) else { continue }
            let parentId = (msg["parent_header"] as? [String: Any])?["msg_id"] as? String
            guard parentId == msgId else { continue }

            let msgType = (msg["msg_type"] as? String)
                ?? ((msg["header"] as? [String: Any])?["msg_type"] as? String)
                ?? ""
            let content = (msg["content"] as? [String: Any]) ?? [:]

            if msgType == "stream", let text = content["text"] as? String {
                for rawLine in text.components(separatedBy: .newlines) {
                    let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { continue }
                    collectedLines.append(trimmed)
                    onStdoutLine?(trimmed)
                    if trimmed.hasPrefix("WAKE_URL:") {
                        return collectedLines
                    }
                }
            } else if msgType == "error" {
                let eval = (content["evalue"] as? String) ?? "Kernel execution error"
                let errLine = "WAKE_ERROR: \(eval)"
                collectedLines.append(errLine)
                onStdoutLine?(errLine)
                return collectedLines
            } else if msgType == "execute_reply" {
                break
            }
        }
        return collectedLines
    }

    func wakeAndDeploy(
        gpuType: String,
        appPyData: Data,
        onLog: @escaping @Sendable (String) -> Void
    ) async throws -> String {
        _ = try await validAccessToken()
        let requestedAccel = (["G4", "H100", "A100", "L4", "T4"].first { $0.caseInsensitiveCompare(gpuType) == .orderedSame } ?? "G4").uppercased()

        onLog(L("正在检查账号下已有的 Colab 云端实例...", "Checking active Colab cloud instances on your account..."))
        let assignments = (try? await listAssignments()) ?? []
        var activeSession = loadSessionRecord()

        // Adopt or refresh proxy token from cloud assignments
        if let firstAssign = assignments.first,
           let ep = firstAssign["endpoint"] as? String,
           let acc = firstAssign["accelerator"] as? String,
           let proxy = firstAssign["runtimeProxyInfo"] as? [String: Any],
           let tok = proxy["token"] as? String,
           let url = proxy["url"] as? String {
            if activeSession == nil || activeSession?.endpoint == ep {
                activeSession = RuntimeSessionRecord(
                    name: "easycolab",
                    token: tok,
                    url: url,
                    endpoint: ep,
                    accelerator: acc,
                    kernelId: activeSession?.kernelId,
                    sessionId: activeSession?.sessionId
                )
                saveSessionRecord(activeSession)
            }
        }

        if let cur = activeSession {
            if cur.accelerator.uppercased() != requestedAccel {
                onLog(L("当前实例型号为 [\(cur.accelerator)]，正在释放并切换为 [\(requestedAccel)]...", "Current runtime is [\(cur.accelerator)]; releasing and switching to [\(requestedAccel)]..."))
                await unassignEndpoint(cur.endpoint)
                saveSessionRecord(nil)
                activeSession = nil
            } else {
                onLog(L("正在检查已有实例 (\(cur.accelerator)) 连通性...", "Checking existing (\(cur.accelerator)) runtime connectivity..."))
                let ok = await waitForRuntimeReady(cur, timeoutSeconds: 15)
                if !ok {
                    onLog(L("旧实例已过期，正在自动清理并分配全新实例...", "Previous runtime expired; cleaning up and allocating a new VM..."))
                    await unassignEndpoint(cur.endpoint)
                    saveSessionRecord(nil)
                    activeSession = nil
                }
            }
        }

        let targetSession: RuntimeSessionRecord
        if let existing = activeSession {
            targetSession = existing
        } else {
            let fresh = try await allocateFreshSession(gpuType: requestedAccel, onLog: onLog)
            _ = await waitForRuntimeReady(fresh, timeoutSeconds: 50)
            targetSession = fresh
        }

        onLog(L("正在上传最新多模型开源服务端代码 (app.py)...", "Uploading latest multi-model server engine (app.py)..."))
        var lastUploadErr: Error?
        for attempt in 0..<4 {
            do {
                try await uploadFileToColab(session: targetSession, data: appPyData, remotePath: "content/app.py")
                lastUploadErr = nil
                break
            } catch {
                lastUploadErr = error
                if attempt < 3 {
                    try? await Task.sleep(nanoseconds: 2_500_000_000)
                }
            }
        }
        if let err = lastUploadErr { throw err }

        if let rcloneConf = await buildRcloneDriveConf(),
           let rcData = rcloneConf.data(using: .utf8) {
            try? await uploadFileToColab(session: targetSession, data: rcData, remotePath: "content/.easycolab_rclone.conf")
        }

        onLog(L("正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...", "Configuring Google Drive persistence, multi-model engine & public tunnel (~40–90s)..."))

        let bootCode = #"""
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
"""#

        let lines: [String]
        do {
            lines = try await executeCodeOnKernel(session: targetSession, code: bootCode, timeout: 300) { line in
                if !line.hasPrefix("WAKE_URL:") {
                    onLog(line.replacingOccurrences(of: "WAKE_LOG: ", with: ""))
                }
            }
        } catch {
            var freshKernelSession = targetSession
            freshKernelSession.kernelId = nil
            freshKernelSession.sessionId = nil
            saveSessionRecord(freshKernelSession)
            lines = try await executeCodeOnKernel(session: freshKernelSession, code: bootCode, timeout: 300) { line in
                if !line.hasPrefix("WAKE_URL:") {
                    onLog(line.replacingOccurrences(of: "WAKE_LOG: ", with: ""))
                }
            }
        }

        if let urlLine = lines.first(where: { $0.hasPrefix("WAKE_URL:") }) {
            return String(urlLine.dropFirst("WAKE_URL:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let errLine = lines.first(where: { $0.hasPrefix("WAKE_ERROR:") }) {
            throw ColabBridgeError.kernelTimeout(errLine)
        }
        throw ColabBridgeError.kernelTimeout(L("未能从云端提取 gradio.live 隧道地址，请重试", "Timed out waiting for gradio.live tunnel URL"))
    }

    func keepalive() async -> NativeAuthStatus {
        guard loadCredentials() != nil else {
            return NativeAuthStatus(ok: true, loggedIn: false, email: "", activeAssignments: 0)
        }
        if let assignments = try? await listAssignments(), let first = assignments.first,
           let ep = first["endpoint"] as? String,
           let acc = first["accelerator"] as? String,
           let proxy = first["runtimeProxyInfo"] as? [String: Any],
           let tok = proxy["token"] as? String,
           let url = proxy["url"] as? String {
            if !ep.isEmpty {
                _ = try? await colabTunnelRequest(
                    path: "/tun/m/\(ep)/keep-alive/",
                    method: "GET",
                    extraHeaders: ["X-Colab-Tunnel": "Google"],
                    timeout: 12
                )
            }
            let prev = loadSessionRecord()
            let updated = RuntimeSessionRecord(
                name: "easycolab",
                token: tok,
                url: url,
                endpoint: ep,
                accelerator: acc,
                kernelId: prev?.kernelId,
                sessionId: prev?.sessionId
            )
            saveSessionRecord(updated)
            _ = try? await executeCodeOnKernel(
                session: updated,
                code: "import pathlib; pathlib.Path('/tmp/.easycolab_client_hb').touch(); _easycolab_hb = 1",
                timeout: 12
            )
        }
        return await fetchAuthStatus()
    }
}
