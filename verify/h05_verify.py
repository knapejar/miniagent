# -*- coding: utf-8 -*-
"""h05: assembler + stack VM, and the three .asm programs.

PRINT is taken to pop its value, JZ/JNZ to pop the condition and
`a b SUB` to compute a - b - the usual stack machine conventions."""
import os

from common import Raises, WS, check, load, report

m = load("vm")
AsmError = getattr(m, "AsmError", None)
VmError = getattr(m, "VmError", None)
check("rules", "AsmError is a ValueError",
      lambda: isinstance(AsmError, type) and issubclass(AsmError, ValueError))


def run(src, **kw):
    return list(m.run(m.assemble(src), **kw))


for label, src, value in [
        ("ADD", "PUSH 2\nPUSH 3\nADD\nPRINT\nHALT", [5]),
        ("SUB", "PUSH 5\nPUSH 3\nSUB\nPRINT\nHALT", [2]),
        ("MUL", "PUSH 6\nPUSH 7\nMUL\nPRINT\nHALT", [42]),
        ("DIV", "PUSH 8\nPUSH 2\nDIV\nPRINT\nHALT", [4]),
        ("MOD", "PUSH 7\nPUSH 3\nMOD\nPRINT\nHALT", [1]),
        ("NEG", "PUSH 5\nNEG\nPRINT\nHALT", [-5]),
        ("EQ", "PUSH 3\nPUSH 3\nEQ\nPRINT\nHALT", [1]),
        ("LT", "PUSH 2\nPUSH 3\nLT\nPRINT\nHALT", [1]),
        ("GT", "PUSH 2\nPUSH 3\nGT\nPRINT\nHALT", [0]),
        ("DUP", "PUSH 4\nDUP\nMUL\nPRINT\nHALT", [16]),
        ("SWAP", "PUSH 1\nPUSH 2\nSWAP\nSUB\nPRINT\nHALT", [1]),
        ("POP", "PUSH 1\nPUSH 2\nPOP\nPRINT\nHALT", [1]),
        ("STORE/LOAD", "PUSH 9\nSTORE 3\nLOAD 3\nPRINT\nHALT", [9]),
        ("slot 15", "PUSH 8\nSTORE 15\nLOAD 15\nPRINT\nHALT", [8]),
        ("comments", "PUSH 1 ; one\n\n; a comment line\nPRINT\nHALT", [1]),
        ("label before an instruction", "start: PUSH 7\nPRINT\nHALT", [7]),
        ("countdown", "PUSH 3\nloop:\nDUP\nPRINT\nPUSH 1\nSUB\nDUP\nJNZ loop\nHALT", [3, 2, 1]),
        ("JZ taken", "PUSH 0\nJZ skip\nPUSH 1\nPRINT\nskip:\nPUSH 2\nPRINT\nHALT", [2]),
        ("JZ not taken", "PUSH 5\nJZ skip\nPUSH 1\nPRINT\nskip:\nPUSH 2\nPRINT\nHALT", [1, 2]),
        ("JMP", "JMP over\nPUSH 1\nPRINT\nover:\nPUSH 2\nPRINT\nHALT", [2]),
        ("CALL/RET", "PUSH 5\nCALL double\nPRINT\nHALT\ndouble:\nPUSH 2\nMUL\nRET", [10]),
        ("nested CALL", "PUSH 3\nCALL f\nPRINT\nHALT\nf:\nCALL g\nPUSH 1\nADD\nRET\n"
                        "g:\nPUSH 10\nMUL\nRET", [31])]:
    check("instructions", label, lambda s=src: run(s), value)

check("errors", "a bad mnemonic raises AsmError", lambda: m.assemble("FOO 1\nHALT"),
      Raises(AsmError or Exception))
check("errors", "a missing label raises AsmError", lambda: m.assemble("JMP nowhere\nHALT"),
      Raises(AsmError or Exception))
check("errors", "stack underflow raises VmError", lambda: run("ADD\nHALT"),
      Raises(VmError or Exception))
check("errors", "division by zero raises VmError", lambda: run("PUSH 1\nPUSH 0\nDIV\nHALT"),
      Raises(VmError or Exception))
check("errors", "an endless loop raises VmError", lambda: run("loop:\nJMP loop", max_steps=1000),
      Raises(VmError or Exception))


def program(name):
    with open(os.path.join(WS, name), encoding="utf-8") as fh:
        return run(fh.read())


FIB0 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
check("programs", "fact.asm prints 3628800", lambda: program("fact.asm"), [3628800])
check("programs", "fib.asm prints the first 15 Fibonacci numbers",
      lambda: program("fib.asm") in (FIB0, FIB0[1:] + [610]))
check("programs", "gcd.asm prints 21", lambda: program("gcd.asm"), [21])
report()
