"""Generate bcrypt password hash for AUTH_PASSWORD_HASH in .env (VPS deployment)."""
import sys
import bcrypt

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_password_hash.py YOUR_PLAIN_PASSWORD")
        print("Output: Add to .env as AUTH_PASSWORD_HASH=<output>")
        sys.exit(1)
    plain = sys.argv[1]
    hashed = bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    print(hashed)

if __name__ == "__main__":
    main()
