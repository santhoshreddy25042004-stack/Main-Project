import os
from dotenv import load_dotenv

# Ensure .env is loaded first
load_dotenv()

from . import (
    set_provider_as_IBMQ,
    set_backend,
    get_random_int32,
    get_random_int64,
    get_random_float,
    get_random_double,
    get_random_complex_rect,
)

def main():
    print("🔑 qRNG — Quantum Random Number Generator (Qiskit 2025+ Compatible)\n")

    token = os.getenv("IBM_API_KEY")
    if token:
        print("✅ Loaded IBM key from .env")
    else:
        token = input("Enter IBM Quantum API token (or press Enter for simulator): ").strip()

    set_provider_as_IBMQ(token if token else None)
    set_backend("ibm_brisbane")

    print("\nChoose random number type:")
    print("1. 32-bit integer")
    print("2. 64-bit integer")
    print("3. Float (0–1)")
    print("4. Double (0–1)")
    print("5. Complex number (rectangular form)")

    choice = input("\nEnter your choice (1–5): ").strip()
    print("\n🎲 Generating quantum random number...\n")

    if choice == "1":
        print(f"🔹 32-bit integer: {get_random_int32()}")
    elif choice == "2":
        print(f"🔹 64-bit integer: {get_random_int64()}")
    elif choice == "3":
        print(f"🔹 Float (0–1): {get_random_float()}")
    elif choice == "4":
        print(f"🔹 Double (0–1): {get_random_double()}")
    elif choice == "5":
        print(f"🔹 Complex: {get_random_complex_rect()}")
    else:
        print("❌ Invalid choice. Please enter 1–5.")


if __name__ == "__main__":
    main()
