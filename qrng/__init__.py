import math
import struct
import os
from dotenv import load_dotenv
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

# =========================================================
# Load environment variables
# =========================================================
load_dotenv()

if os.getenv("IBM_API_KEY"):
    print("✅ Loaded IBM Quantum API key from .env")
else:
    print("⚠️ No IBM_API_KEY found in .env; will use simulator fallback.")

# =========================================================
# 🌐 Globals
# =========================================================
service = None
_backend = None
_circuit = None
_bitCache = ""
VERBOSE = False   # set True if you want debug logs


def _log(msg: str):
    if VERBOSE:
        print(msg)


# =========================================================
# ✅ Provider Setup
# =========================================================
def set_provider_as_IBMQ(token: str | None = None, instance: str | None = None):
    """Initialize IBM Quantum Runtime Service."""
    global service
    try:
        if not token:
            raise ValueError("No token provided")

        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=token,
            instance=instance
        )
        _log("✅ Connected to IBM Quantum Runtime Service.")

    except Exception as e:
        _log(f"⚠️ IBM Quantum connection failed ({e}); using simulator.")
        service = None


# =========================================================
# ✅ Backend Setup (INTERACTIVE)
# =========================================================
def set_backend():
    """
    Show available IBM Quantum backends and allow user selection.
    Falls back to Aer simulator if unavailable or skipped.
    """
    global _backend, service

    try:
        if service is None:
            raise RuntimeError("IBM service not available")

        backends = service.backends()

        if not backends:
            raise RuntimeError("No IBM backends available for this account")

        print("\n🔍 Available IBM Quantum Backends:")
        for i, b in enumerate(backends, start=1):
            print(f"{i}. {b.name}")

        choice = input(
            "\n👉 Select backend number "
            "(or press Enter for simulator): "
        ).strip()

        if not choice:
            raise RuntimeError("User selected simulator")

        idx = int(choice) - 1
        if idx < 0 or idx >= len(backends):
            raise ValueError("Invalid backend selection")

        _backend = backends[idx]
        _log(f"✅ Using IBM Quantum backend: {_backend.name}")

    except Exception as e:
        _log(f"⚠️ Backend selection failed ({e}); using Aer simulator.")
        from qiskit_aer import AerSimulator
        _backend = AerSimulator()
        _log("✅ Using local qasm_simulator backend.")

    _create_circuit(4)


# =========================================================
# ✅ Circuit Definition
# =========================================================
def _create_circuit(n: int):
    global _circuit
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n)
    qc = QuantumCircuit(qr, cr)
    qc.h(qr)
    qc.measure(qr, cr)
    _circuit = qc


# =========================================================
# ✅ Quantum / Simulator Bit Request
# =========================================================
def _request_bits(n: int):
    global _bitCache, _backend, _circuit

    # --- IBM Runtime Sampler (if hardware) ---
    if service is not None and _backend is not None and "ibm" in _backend.name:
        try:
            from qiskit_ibm_runtime import Sampler
            sampler = Sampler(backend=_backend)
            job = sampler.run([_circuit])
            result = job.result()
            counts = result[0].data.meas.get_counts()
            bits = max(counts, key=counts.get)
            _bitCache += bits
            return
        except Exception:
            pass  # silently fall back to Aer

    # --- Aer Simulator fallback ---
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    sim = AerSimulator()
    compiled = transpile(_circuit, sim)
    job = sim.run(compiled, shots=n)
    counts = job.result().get_counts()
    bits = max(counts, key=counts.get)
    _bitCache += bits


# =========================================================
# ✅ Public API
# =========================================================
def get_bit_string(n: int) -> str:
    global _bitCache
    while len(_bitCache) < n:
        _request_bits(4)
    bits = _bitCache[:n]
    _bitCache = _bitCache[n:]
    return bits


def get_random_int(min_val: int, max_val: int) -> int:
    delta = max_val - min_val
    bits = math.floor(math.log2(delta)) + 1
    val = int(get_bit_string(bits), 2)
    while val > delta:
        val = int(get_bit_string(bits), 2)
    return val + min_val


def get_random_int32() -> int:
    return int(get_bit_string(32), 2)


def get_random_int64() -> int:
    return int(get_bit_string(64), 2)


def get_random_float(min_val=0.0, max_val=1.0) -> float:
    unpacked = 0x3F800000 | (get_random_int32() >> 9)
    packed = struct.pack("I", unpacked)
    value = struct.unpack("f", packed)[0] - 1.0
    return (max_val - min_val) * value + min_val


def get_random_double(min_val=0.0, max_val=1.0) -> float:
    unpacked = 0x3FF0000000000000 | (get_random_int64() >> 12)
    packed = struct.pack("Q", unpacked)
    value = struct.unpack("d", packed)[0] - 1.0
    return (max_val - min_val) * value + min_val


def get_random_complex_rect(real_min=0, real_max=1, imag_min=None, imag_max=None):
    if imag_min is None:
        imag_min = real_min
    if imag_max is None:
        imag_max = real_max
    return (
        get_random_float(real_min, real_max)
        + get_random_float(imag_min, imag_max) * 1j
    )


def get_random_complex_polar(r=1, theta=2 * math.pi):
    r0 = r * math.sqrt(get_random_float(0, 1))
    theta0 = get_random_float(0, theta)
    return r0 * math.cos(theta0) + r0 * math.sin(theta0) * 1j

