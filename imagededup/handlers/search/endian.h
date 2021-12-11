/* Endianness detection and swapping (v3)
 * Portable Snippets - https://github.com/nemequ/portable-snippets
 * Created by Evan Nemerson <evan@nemerson.com>
 *
 *   To the extent possible under law, the authors have waived all
 *   copyright and related or neighboring rights to this code.  For
 *   details, see the Creative Commons Zero 1.0 Universal license at
 *   https://creativecommons.org/publicdomain/zero/1.0/
 */

#if !defined(PSNIP_ENDIAN_H)
#define PSNIP_ENDIAN_H

/* For maximum portability include the exact-int module from
   portable snippets. */
#if \
  !defined(psnip_uint64_t) || \
  !defined(psnip_uint32_t) || \
  !defined(psnip_uint16_t)
#  include <stdint.h>
#  if !defined(psnip_uint64_t)
#    define psnip_uint64_t uint64_t
#  endif
#  if !defined(psnip_uint32_t)
#    define psnip_uint32_t uint32_t
#  endif
#  if !defined(psnip_uint16_t)
#    define psnip_uint16_t uint16_t
#  endif
#endif

#if !defined(PSNIP_BUILTIN_H)
#  include "../builtin/builtin.h"
#endif

#if !defined(PSNIP_ENDIAN_STATIC_INLINE)
#  if defined(__GNUC__)
#    define PSNIP_ENDIAN__COMPILER_ATTRIBUTES __attribute__((__unused__))
#  else
#    define PSNIP_ENDIAN__COMPILER_ATTRIBUTES
#  endif

#  if defined(HEDLEY_INLINE)
#    define PSNIP_ENDIAN__INLINE HEDLEY_INLINE
#  elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#    define PSNIP_ENDIAN__INLINE inline
#  elif defined(__GNUC_STDC_INLINE__)
#    define PSNIP_ENDIAN__INLINE __inline__
#  elif defined(_MSC_VER) && _MSC_VER >= 1200
#    define PSNIP_ENDIAN__INLINE __inline
#  else
#    define PSNIP_ENDIAN__INLINE
#  endif

#  define PSNIP_ENDIAN__FUNCTION PSNIP_ENDIAN__COMPILER_ATTRIBUTES static PSNIP_ENDIAN__INLINE
#endif

#if !defined(PSNIP_ENDIAN_LITTLE)
#  define PSNIP_ENDIAN_LITTLE 1234
#endif
#if !defined(PSNIP_ENDIAN_BIG)
#  define PSNIP_ENDIAN_BIG 4321
#endif
#if !defined(PSNIP_ENDIAN_PDP)
#  define PSNIP_ENDIAN_PDP 3412
#endif

/* Detection
 *
 * Detecting endianness can be a bit tricky.  There isn't really a
 * good standard way of determining endianness, and it's actually
 * possible to mix endianness within a single program.  This is
 * currently pretty rare, though.
 *
 * We try to define PSNIP_ENDIAN_ORDER to PSNIP_ENDIAN_LITTLE,
 * PSNIP_ENDIAN_BIG, or PSNIP_ENDIAN_PDP.  Additionally, you can use
 * the PSNIP_RT_BYTE_ORDER to check the runtime byte order, which is a
 * bit more reliable (though it may introduce some runtime overhead).
 *
 * In the event we are unable to determine endianness at compile-time,
 * PSNIP_ENDIAN_ORDER is left undefined and you will be forced to rely
 * on PSNIP_RT_BYTE_ORDER. */

#if !defined(PSNIP_ENDIAN_FORCE_RT)

#if !defined(PSNIP_ENDIAN_ORDER)
/* GCC (and compilers masquerading as GCC) define  __BYTE_ORDER__. */
#  if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#    define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_LITTLE
#  elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#    define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_BIG
#  elif defined(__BYTE_ORDER__) && defined(__ORDER_PDP_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_PDP_ENDIAN__)
#    define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_PDP
/* TI defines _BIG_ENDIAN or _LITTLE_ENDIAN */
#  elif defined(_BIG_ENDIAN)
#    define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_BIG
#  elif defined(_LITTLE_ENDIAN)
#    define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_LITTLE
/* We know the endianness of some common architectures.  Common
 * architectures not listed (ARM, POWER, MIPS, etc.) here are
 * bi-endian. */
#  elif defined(__amd64) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#    define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_LITTLE
#  elif defined(__s390x__) || defined(__zarch__)
#    define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_BIG
/* Looks like we'll have to rely on the platform.  If we're missing a
 * platform, please let us know. */
#  elif defined(_WIN32)
#    define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_LITTLE
#  elif defined(sun) || defined(__sun) /* Solaris */
#    include <sys/byteorder.h>
#    if defined(_LITTLE_ENDIAN)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_LITTLE
#    elif defined(_BIG_ENDIAN)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_BIG
#    endif
#  elif defined(__APPLE__)
#    include <libkern/OSByteOrder.h>
#    if defined(__LITTLE_ENDIAN__)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_LITTLE
#    elif defined(__BIG_ENDIAN__)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_BIG
#    endif
#  elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__) || defined(BSD)
#    include <machine/endian.h>
#    if defined(__BYTE_ORDER) && (__BYTE_ORDER == __LITTLE_ENDIAN)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_LITTLE
#    elif defined(__BYTE_ORDER) && (__BYTE_ORDER == __BIG_ENDIAN)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_BIG
#    elif defined(__BYTE_ORDER) && (__BYTE_ORDER == __PDP_ENDIAN)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_PDP
#    endif
#  elif defined(__linux__) || defined(__linux) || defined(__gnu_linux__)
#    include <endian.h>
#    if defined(__BYTE_ORDER) && defined(__LITTLE_ENDIAN) && (__BYTE_ORDER == __LITTLE_ENDIAN)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_LITTLE
#    elif defined(__BYTE_ORDER) && defined(__BIG_ENDIAN) && (__BYTE_ORDER == __BIG_ENDIAN)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_BIG
#    elif defined(__BYTE_ORDER) && defined(__PDP_ENDIAN) && (__BYTE_ORDER == __PDP_ENDIAN)
#      define PSNIP_ENDIAN_ORDER PSNIP_ENDIAN_PDP
#    endif
#  endif
#endif

/* PDP endian not yet supported.  Patches welcome. */
#if defined(PSNIP_ENDIAN_ORDER) && (PSNIP_ENDIAN_ORDER == PSNIP_ENDIAN_PDP)
#  error PDP endian is not supported.
#endif

#endif /* !defined(PSNIP_ENDIAN_FORCE_RT) */

static const union {
  unsigned long long value;
  unsigned char bytes[4];
} psnip_endian_rt_data = {
  1
};

#define PSNIP_ENDIAN_ORDER_RT_IS_LE (psnip_endian_rt_data.bytes[0] == 1)
#define PSNIP_ENDIAN_ORDER_RT_IS_BE (psnip_endian_rt_data.bytes[sizeof(unsigned long long) - 1] == 1)
#define PSNIP_ENDIAN_ORDER_RT (PSNIP_ENDIAN_ORDER_RT_IS_LE ? PSNIP_ENDIAN_LITTLE : PSNIP_ENDIAN_BIG)

#if defined(PSNIP_ENDIAN_FORCE_RT) || !defined(PSNIP_ENDIAN_ORDER)
#define PSNIP_ENDIAN__DEFINE_LE_FUNC(siz) \
  PSNIP_ENDIAN__FUNCTION \
  psnip_uint##siz##_t psnip_endian_le##siz(psnip_uint##siz##_t v) { \
    return PSNIP_ENDIAN_ORDER_RT_IS_LE ? v : psnip_builtin_bswap##siz(v); \
  }

#define PSNIP_ENDIAN__DEFINE_BE_FUNC(siz) \
  PSNIP_ENDIAN__FUNCTION \
  psnip_uint##siz##_t psnip_endian_be##siz(psnip_uint##siz##_t v) { \
    return PSNIP_ENDIAN_ORDER_RT_IS_BE ? v : psnip_builtin_bswap##siz(v); \
  }

PSNIP_ENDIAN__DEFINE_LE_FUNC(16)
PSNIP_ENDIAN__DEFINE_LE_FUNC(32)
PSNIP_ENDIAN__DEFINE_LE_FUNC(64)
PSNIP_ENDIAN__DEFINE_BE_FUNC(16)
PSNIP_ENDIAN__DEFINE_BE_FUNC(32)
PSNIP_ENDIAN__DEFINE_BE_FUNC(64)
#elif PSNIP_ENDIAN_ORDER == PSNIP_ENDIAN_LITTLE
#  define psnip_endian_le16(v) ((psnip_uint16_t) (v))
#  define psnip_endian_le32(v) ((psnip_uint32_t) (v))
#  define psnip_endian_le64(v) ((psnip_uint64_t) (v))
#  define psnip_endian_be16(v) psnip_builtin_bswap16((psnip_uint16_t) (v))
#  define psnip_endian_be32(v) psnip_builtin_bswap32((psnip_uint32_t) (v))
#  define psnip_endian_be64(v) psnip_builtin_bswap64((psnip_uint64_t) (v))
#elif PSNIP_ENDIAN_ORDER == PSNIP_ENDIAN_BIG
#  define psnip_endian_le16(v) psnip_builtin_bswap16((psnip_uint16_t) (v))
#  define psnip_endian_le32(v) psnip_builtin_bswap32((psnip_uint32_t) (v))
#  define psnip_endian_le64(v) psnip_builtin_bswap64((psnip_uint64_t) (v))
#  define psnip_endian_be16(v) ((psnip_uint16_t) (v))
#  define psnip_endian_be32(v) ((psnip_uint32_t) (v))
#  define psnip_endian_be64(v) ((psnip_uint64_t) (v))
#else
#  error Unknown endianness.
#endif

#endif /* !defined(PSNIP_ENDIAN_H) */
