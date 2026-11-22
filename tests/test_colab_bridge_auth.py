import json
import os
import tempfile
import unittest
from unittest import mock

import colab_bridge


def _sample_token_payload(client_id: str = "1234567890-custom.apps.googleusercontent.com") -> dict:
    return {
        "token": "ya29.test_access_token",
        "refresh_token": "1//test_refresh_token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": client_id,
        "client_secret": "test-secret",
        "scopes": list(colab_bridge.PUBLIC_SCOPES),
        "expiry": "2099-01-01T00:00:00Z",
    }


class ColabBridgeKeyringAuthTest(unittest.TestCase):
    def test_keyring_save_load_delete_and_legacy_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy_path = os.path.join(tmp, "colab-cli", "token.json")
            os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
            with open(legacy_path, "w", encoding="utf-8") as f:
                json.dump(_sample_token_payload(), f)

            store: dict[str, str] = {}

            def fake_get():
                return store.get("oauth_token")

            def fake_save(val: str):
                store["oauth_token"] = val

            def fake_remove_legacy():
                if os.path.exists(legacy_path):
                    os.remove(legacy_path)

            def fake_delete(svc: str, usr: str):
                store.pop(usr, None)

            fake_keyring = mock.MagicMock()
            fake_keyring.delete_password.side_effect = fake_delete

            with (
                mock.patch.object(colab_bridge, "TOKEN_CONFIG_PATH", legacy_path),
                mock.patch.object(colab_bridge, "_USE_KEYRING_AUTH", True),
                mock.patch.object(colab_bridge._colab_auth, "_get_token_from_keyring", side_effect=fake_get, create=True),
                mock.patch.object(colab_bridge._colab_auth, "_save_token_to_keyring", side_effect=fake_save, create=True),
                mock.patch.object(colab_bridge._colab_auth, "_remove_legacy_token", side_effect=fake_remove_legacy, create=True),
                mock.patch.object(colab_bridge._colab_auth, "_run_with_timeout", side_effect=lambda fn, *a, **kw: fn(*a), create=True),
                mock.patch.dict("sys.modules", {"keyring": fake_keyring}),
            ):
                # Falls back to legacy file when keyring is empty
                self.assertTrue(colab_bridge._has_valid_token())

                # Saving stores in keyring and removes legacy plaintext file
                payload_json = json.dumps(_sample_token_payload())
                colab_bridge._save_oauth_token_json(payload_json)
                self.assertEqual(store.get("oauth_token"), payload_json)
                self.assertFalse(os.path.exists(legacy_path))
                self.assertTrue(colab_bridge._has_valid_token())

                # Default _build_rclone_drive_conf() reads token from keyring
                conf = colab_bridge._build_rclone_drive_conf()
                self.assertIn("[easycolab_gdrive]", conf)
                self.assertIn("1234567890-custom.apps.googleusercontent.com", conf)

                # Deleting clears keyring entry so token is no longer present
                colab_bridge._delete_oauth_token()
                self.assertEqual(colab_bridge._load_oauth_token_json(), "")
                self.assertFalse(colab_bridge._has_valid_token())

    def test_fallback_to_file_storage_when_keyring_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok_path = os.path.join(tmp, "nested", "token.json")
            payload_json = json.dumps(_sample_token_payload())

            with (
                mock.patch.object(colab_bridge, "TOKEN_CONFIG_PATH", tok_path),
                mock.patch.object(colab_bridge, "_USE_KEYRING_AUTH", True),
                mock.patch.object(
                    colab_bridge._colab_auth,
                    "_save_token_to_keyring",
                    side_effect=RuntimeError("keyring locked"),
                    create=True,
                ),
                mock.patch.object(
                    colab_bridge._colab_auth,
                    "_get_token_from_keyring",
                    side_effect=RuntimeError("keyring locked"),
                    create=True,
                ),
            ):
                colab_bridge._save_oauth_token_json(payload_json)
                self.assertTrue(os.path.isfile(tok_path))
                self.assertEqual(colab_bridge._load_oauth_token_json(), payload_json)
                self.assertTrue(colab_bridge._has_valid_token())

                colab_bridge._delete_oauth_token()
                self.assertFalse(os.path.exists(tok_path))

    def test_ensure_colab_config_dir_creates_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = os.path.join(tmp, "fresh-config", "colab-cli")
            tok_path = os.path.join(missing_dir, "token.json")
            self.assertFalse(os.path.exists(missing_dir))
            with mock.patch.object(colab_bridge, "TOKEN_CONFIG_PATH", tok_path):
                colab_bridge._ensure_colab_config_dir()
            self.assertTrue(os.path.isdir(missing_dir))


if __name__ == "__main__":
    unittest.main()
