# Built-Ins

This module provides portable implementations of many compiler
builtins and intrinsics, allowing you to use builtins and intrinsics
on compilers which don't support them.  This includes other compilers
(*e.g.*, GCC builtins on MSVC) and older versions of the same
compiler.

We also provide exact-width variants of many builtins; no more calling
different functions depending on the size of `int`, `long`, `long
long`, etc.  These are typically just aliases for the appropriate
function, but if we can't find an appropriate type a portable
implementation will be used.

If you define `PSNIP_BUILTIN_EMULATE_NATIVE` *before* `builtin.h` is
included, this header will also define any missing native-style
built-ins, allowing you to use the native names without regard for
which compiler is actually in use (*i.e.*, you can use `__builtin_ffs`
directly in MSVC, or any other compiler, including GCC < 3.3).

If the compiler already has the builtin, the psnip function will
simply be defined to that builtin (*e.g.*,
`#define psnip_builtin_clz __builtin_clz`).  If the compiler does not
have an implementation one will be provided using either a
built-in/intrinsic the compiler *does* support (*e.g.*, using an MSVC
intrinsic to implement a GCC built-in), inline assembly,
architecture-specific functions, or a fully-portable pure C
implementation.

For example, for GCC's `__builtin_ffs` builtin, we provide
implementations which work everywhere (including versions of GCC prior
to 3.3, when `__builtin_ffs` was introduced) of the following
functions:

```c
int psnip_builtin_ffs(int);
int psnip_builtin_ffsl(long);
int psnip_builtin_ffsll(long long);
int psnip_builtin_ffs32(psnip_int32_t);
int psnip_builtin_ffs64(psnip_int64_t);
```

Additionally, when when PSNIP_BUILTIN_EMULATE_NATIVE is defined (and
the compiler doesn't already provide them), we also provide

```c
int __builtin_ffs(int);
int __builtin_ffsl(long);
int __builtin_ffsll(long long);
```

Note that these are often provided as macros, the prototypes are for
documentation only.

## Dependencies

To maximize portability you should #include the exact-int module
before including builtin.h, but if you don't want to add the extra
file to your project you can omit it and this module will simply rely
on <stdint.h>.  As an alternative you may define the following macros
to appropriate values yourself:

 * `psnip_int8_t`
 * `psnip_uint8_t`
 * `psnip_int16_t`
 * `psnip_uint16_t`
 * `psnip_int32_t`
 * `psnip_uint32_t`
 * `psnip_int64_t`
 * `psnip_uint64_t`

## Implementation Status

Virtually every generic builtin we can implement has been implemented.
This should work almost everywhere, but every commit is tested before
landing in the master branch on various versions of GCC, clang, MSVC,
and PGI (thanks to [Travis
CI](https://travis-ci.org/nemequ/portable-snippets) and
[AppVeyor](https://ci.appveyor.com/project/quixdb/portable-snippets)).
Sporadic testing is also done on ICC and Oracle Developer Studio.

GCC builtins:

 - [x] ffs, ffsl, ffsll, ffs32, ffs64
 - [x] clz, clzl, clzll, clz32, clz64
 - [x] ctz, ctzl, ctzll, ctz32, ctz64
 - [x] clrsb, clrsbl, clrsbll, clrsb32, clrsb64
 - [x] popcount, popcountl, popcountll, popcount32, popcount64
 - [x] parity, parityl, parityll, parity32, parity64
 - [x] bswap16, bswap32, bswap64

Clang builtins:

 - [x] bitreverse8, bitreverse16, bitreverse32, bitreverse64
 - [x] addcb, addcs, addc, addcl, addcll, addc8, addc16, addc32, addc64
 - [x] subcb, subcs, subc, subcl, subcll, subc8, subc16, subc32, subc64

MSVC intrinsics:

 - [x] rotl8, rotl16, rotl, rotl64
 - [x] rotr8, rotr16, rotr, rotr64
 - [x] BitScanForward, BitScanForward64
 - [x] BitScanReverse, BitScanReverse64
 - [ ] mul128, umul128
 - [x] shiftleft128, shiftright128
 - [ ] mulh, umulh
 - [x] byteswap_ushort, byteswap_ulong, byteswap_uint64
 - [x] bittest, bittest64
 - [x] bittestandcomplement, bittestandcomplement64
 - [x] bittestandreset, bittestandreset64
 - [x] bittestandset, bittestandset64

If we are missing a function you feel should be included, please [file
an issue](https://github.com/nemequ/portable-snippets/issues).  Please
keep in mind that some things are simply impossible to implement
without compiler support.

## Alternatives & Supplements

For overflow-safe integer operations (i.e., `__builtin_*_overflow`),
use [safe-math.h](../safe-math).

For bswap/byteswap functions, you should really use
[endian.h](../endian), which depends on this module and handles
endianness detection as well as providing easier to use APIs which
integrate endianness detection logic.

For SIMD intrinsics (SSE, AVX, NEON, etc.), take a look at the
[SIMDe](https://github.com/nemequ/simde/) project.

For things which are effectively compiler hints (such as
`__builtin_expect`) as opposed to data manipulation functions, see
[Hedley](https://nemequ.github.io/hedley/).

## Areas for future work

### Optimization

Performance should be pretty good but we're always open to shaving off
a few operations, even if it means creating different variants for
different compilers or architectures.

Creating implementations of one compiler's builtins using builtins
from another is probably your best bet to improve performance.
Another useful possibility is using architecure-specific builtins, or
even embedded assembly, when they are available.

### Architecture-specific builtins

GCC and MSVC both have lots of architecture-specific builtins.
Especially GCC, which supports many architectures.  If you come across
one which is useful and could be implemented with a portable fallback,
let us know.

### Builtins from other compilers

We've looked at GCC, MSVC, and Clang, but we're happy to support
builtins from other compilers, too.
