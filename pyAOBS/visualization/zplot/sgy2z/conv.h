/******************************************************************************/
/* All source code copyright Cambridge University (c) 1989,1990               */
/*----------------------------------------------------------------------------*/
/* Author:         Andrew Scholan                                             */
/*============================================================================*/
/* Filename:       conv.h                                                     */
/* Version:        3.31                                                       */
/* Compiler:       Any ANSI Compatible C Compiler.                            */
/* Last Edit:      05:20pm on 08/03/93                          Edit Ref: 2   */
/*----------------------------------------------------------------------------*/
/* SYNOPSIS:                                                                  */
/*   Low level conversion routines for "ffc".                                 */
/******************************************************************************/

void ebcdic_to_ascii() ;
  /* This converts the ascii format source character to an EBCDIC format
  destination character. */

void ascii_to_ebcdic() ;
  /* This converts the EBCDIC source to an ASCII destination. Characters
  which cannot be converted are converted to spaces. */

void byte_swop_2 () ;
  /* This flips the two bytes in a 2 byte integer pointed to by src and returns
  two bytes in dst */

void byte_swop_4 () ;
  /* This is as above but for a four byte integer */

void byte_swop_8 () ;
  /* This is as above but for a 8 byte integer */

void short_to_long () ;
  /* This converts a little-endian 2 byte integer to a little-endian 4 byte
  integer. The integer is assumed to be signed.*/

void short_to_long_uns () ;
  /* This is as above but assumes an unsigned integer */

void long_to_short () ;
  /* This converts a little endian signed integer of 4 bytes to one of two. If
  the resulting integer cannot be fitted in 2 bytes the value is forced to
  -32768 or +32767 according to the sign of the original integer. */

void long_to_short_uns () ;
  /* As above but for unsigned integers. */

void short_s_to_uns () ;
  /* Converts signed little endian short to unsigned little endian short */

void short_uns_to_s () ;
  /* Converse of above */

void long_s_to_uns () ;
  /* Converts signed little endian longs to unsigned little endian longs */

void long_uns_to_s () ;
  /* Converse of above. */

void nil_conv () ;
  /* Transfers "sz*n" bytes from the source buffer to the destination buffer */

void long_uns_to_ieee32() ;
  /* Converts an unsigned 4 byte integer to a 32 bit ieee floating point real.*/

void long_s_to_ieee32() ;
  /* Converts a signed 4 byte integer to a 32 bit ieee floating point real. */

void long_uns_to_ieee64() ;
  /* Converts an unsigned 4 byte integer to a 64 bit ieee floating point real.*/

void long_s_to_ieee64() ;
  /* Converts a signed 4 byte integer to a 64 bit ieee floating point real. */

void long_uns_to_convex32() ;
  /* Converts an unsigned 4 byte integer to a 32 bit convex floating point real.*/

void long_s_to_convex32() ;
  /* Converts a signed 4 byte integer to a 32 bit convex floating point real. */

void long_uns_to_convex64() ;
  /* Converts an unsigned 4 byte integer to a 64 bit convex floating point real.*/

void long_s_to_convex64() ;
  /* Converts a signed 4 byte integer to a 64 bit convex floating point real. */

void long_uns_to_ibm32() ;
  /* Converts an unsigned 4 byte integer to a 32 bit IBM floating point real. */

void long_s_to_ibm32() ;
  /* Converts a signed 4 byte integer to a 32 bit IBM floating point real. */

void long_uns_to_ibm64() ;
  /* Does bugger all */

void long_s_to_ibm64() ;
  /* Again Bugger All */

void single_to_double() ;
  /* This procedure converts a 32 bit fp number to a 64 bit fp number in a
  given format. If a number cannot be converted it is returned as zero. */

void double_to_single() ;
  /* This procedure converts a 64 bit real to a 32 bit real. If the conversion
  cannot be done the return value is zero. */

void single_to_double_ibm() ;
  /* Does bugger all */

void double_to_single_ibm() ;
  /* Does bugger all */

void ieee_to_convex_32() ;
  /* Converts IEEE 32 bit real to Convex 32 bit real. */

void ieee_to_convex_64() ;
  /* Converts IEEE 64 bit real to Convex 64 bit real. */

void convex_to_ieee_32() ;
  /* Converts Convex 32 bit real to IEEE 32 bit real. */

void convex_to_ieee_64() ;
  /* Converts Convex 64 bit real to IEEE 64 bit real. */

void ibm_to_convex_32() ;
  /* Converts ibm to convex style floating point numbers */

void ibm_to_ieee_32() ;
  /* Converts ibm to ieee style floating point numbers */

void convex_to_ibm_32() ;
  /* Converts convex to ibm style floating point numbers */

void ieee_to_ibm_32() ;
  /* Converts ieee to ibm style floating point numbers */

void ibm_to_convex_64() ;
  /* Does bugger all */

void ibm_to_ieee_64() ;
  /* Does bugger all */

void convex_to_ibm_64() ;
  /* Does bugger all */

void ieee_to_ibm_64() ;
  /* Does bugger all */
