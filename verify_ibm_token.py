"""
verify_ibm_token.py (Qiskit 2025+ version)
Auto-detects correct IBM Quantum channel and verifies your token.
"""

from qiskit_ibm_runtime import QiskitRuntimeService

def verify_token(token: str):
    """Try both IBM channels ('ibm_cloud' and 'ibm_quantum_platform')."""
    for channel in ["ibm_cloud", "ibm_quantum_platform"]:
        print(f"\n🔍 Trying channel: {channel} ...")
        try:
            service = QiskitRuntimeService(channel=channel, token=token)
            backends = service.backends()
            if backends:
                print(f"✅ Success! Connected via '{channel}' ({len(backends)} backends found).")
                service.save_account(channel=channel, token=token, overwrite=True)
                print("💾 Credentials saved for future use.")
                return
        except Exception as e:
            print(f"⚠️ Channel '{channel}' failed: {e}")
    print("❌ Token verification failed for both channels.")

if __name__ == "__main__":
    print("🔑 IBM Quantum Token Verifier (Qiskit 2025+ Compatible)")
    token = input("Enter your IBM Quantum API token: ").strip()
    verify_token(token)
