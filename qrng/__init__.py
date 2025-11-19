"""
qRNG — Quantum Random Number Generator (Qiskit 2025+ Compatible)
Author: Sai Sreekar + ChatGPT (2025)
---------------------------------------------------------------
• Uses IBM Quantum (SamplerV2 + Qiskit Runtime Service) for true
  quantum random bit generation.
• Automatically falls back to local Aer simulator if IBM hardware
  is unavailable or plan restrictions prevent real-device use.
---------------------------------------------------------------
"""

import math
import struct
import os
from dotenv import load_dotenv
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

# Load environment variables (IBM_API_KEY)
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
VERBOSE = True   # 👈 set False to silence logs


def _log(msg: str):
    """Helper: print only if VERBOSE=True."""
    if VERBOSE:
        print(msg)


# =========================================================
# ✅ Provider Setup
# =========================================================
def set_provider_as_IBMQ(token: str | None = None, instance: str | None = None):
    """Initialize IBM Quantum Service; fallback to Aer if unavailable."""
    global service
    try:
        if token:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)
            _log("✅ Connected to IBM Quantum via Qiskit Runtime Service.")
        else:
            raise ValueError("No token provided")
    except Exception as e:
        _log(f"⚠️ IBM Quantum connection failed ({e}); using local simulator.")
        service = None


# =========================================================
# ✅ Backend Setup
# =========================================================
def set_backend(backend: str = "ibm_brisbane"):
    """Select IBM Quantum backend or Aer simulator."""
    global _backend, service
    try:
        if service is not None:
            _backend = service.backend(backend)
            _log(f"✅ Using IBM Quantum backend: {_backend.name}")
        else:
            raise RuntimeError("No IBM service available")
    except Exception as e:
        _log(f"⚠️ Backend error ({e}); switching to Aer simulator.")
        from qiskit_aer import AerSimulator
        _backend = AerSimulator()
        _log("✅ Using local qasm_simulator backend.")

    _create_circuit(4)  # default circuit size


# =========================================================
# ✅ Circuit Definition
# =========================================================
def _create_circuit(n: int):
    """Create Hadamard + Measure circuit with n qubits."""
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
    """Generate random bits using IBM Sampler or Aer fallback."""
    global _bitCache, _backend, _circuit, service
    try:
        from qiskit_ibm_runtime import SamplerV2 as Sampler, Session

        _log(f"🎯 Requesting {n}-bit quantum sample from '{_backend.name}'...")
        with Session(backend=_backend) as session:
            sampler = Sampler(session=session)
            job = sampler.run([_circuit])
            result = job.result()
            counts = result[0].data.meas.get_counts()
            bits = max(counts, key=counts.get)
            _bitCache += bits
            _log(f"✅ Quantum sample received ({len(bits)} bits).")

    except Exception as e:
        _log(f"⚠️ Sampler failed ({e}); switching to Aer fallback.")
        try:
            from qiskit_aer import AerSimulator
            from qiskit import transpile
            sim = AerSimulator()
            compiled = transpile(_circuit, sim)
            job = sim.run(compiled, shots=n)
            counts = job.result().get_counts()
            bits = max(counts, key=counts.get)
            _bitCache += bits
            _log(f"✅ Aer fallback successful ({len(bits)} bits).")
        except Exception as err:
            raise RuntimeError(f"❌ Aer fallback failed too: {err}")


# =========================================================
# ✅ Public API
# =========================================================
def get_bit_string(n: int) -> str:
    """Return an n-bit random bitstring."""
    global _bitCache
    while len(_bitCache) < n:
        _request_bits(4)
    bits = _bitCache[:n]
    _bitCache = _bitCache[n:]
    return bits


def get_random_int(min_val: int, max_val: int) -> int:
    """Return random integer [min_val, max_val]."""
    delta = max_val - min_val
    n = math.floor(math.log2(delta)) + 1
    val = int(get_bit_string(n), 2)
    while val > delta:
        val = int(get_bit_string(n), 2)
    return val + min_val


def get_random_int32() -> int:
    """Return a random 32-bit integer."""
    return int(get_bit_string(32), 2)


def get_random_int64() -> int:
    """Return a random 64-bit integer."""
    return int(get_bit_string(64), 2)


def get_random_float(min_val=0.0, max_val=1.0) -> float:
    """Return a random float [min_val, max_val]."""
    unpacked = 0x3F800000 | get_random_int32() >> 9
    packed = struct.pack("I", unpacked)
    value = struct.unpack("f", packed)[0] - 1.0
    return (max_val - min_val) * value + min_val


def get_random_double(min_val=0.0, max_val=1.0) -> float:
    """Return a random double [min_val, max_val]."""
    unpacked = 0x3FF0000000000000 | get_random_int64() >> 12
    packed = struct.pack("Q", unpacked)
    value = struct.unpack("d", packed)[0] - 1.0
    return (max_val - min_val) * value + min_val


def get_random_complex_rect(real_min=0, real_max=1, imag_min=None, imag_max=None):
    """Return a random complex number (rectangular form)."""
    if imag_min is None:
        imag_min = real_min
    if imag_max is None:
        imag_max = real_max
    re = get_random_float(real_min, real_max)
    im = get_random_float(imag_min, imag_max)
    return re + im * 1j


def get_random_complex_polar(r=1, theta=2 * math.pi):
    """Return a random complex number (polar form)."""
    r0 = r * math.sqrt(get_random_float(0, 1))
    theta0 = get_random_float(0, theta)
    return r0 * math.cos(theta0) + r0 * math.sin(theta0) * 1j
