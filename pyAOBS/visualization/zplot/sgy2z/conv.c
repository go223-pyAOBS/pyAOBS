/******************************************************************************/
/* All source code copyright Cambridge University (c) 1989,1990               */
/*----------------------------------------------------------------------------*/
/* Author:         Andrew Scholan                                             */
/*============================================================================*/
/* Filename:       conv.c                                                     */
/* Version:        3.31                                                       */
/* Compiler:       Any ANSI Compatible C Compiler.                            */
/* Last Edit:      05:24pm on 08/03/93                          Edit Ref: 8   */
/*----------------------------------------------------------------------------*/
/* SYNOPSIS:                                                                  */
/*   Low level conversion programs for "ffc".                                 */
/******************************************************************************/

#include <stdlib.h>
#include "conv.h"

#define ibm_exp_32        64
#define ibm_exp_64       256
#define convex_exp_32    129
#define convex_exp_64   1025
#define ieee_exp_32      127
#define ieee_exp_64     1023
#define const

/** Identification string *****************************************************/

const char conv_identifier_[] = "conv.c       3.31 05:24pm 08/03/93    8" ;

/* This look-up table converts 7 bit ascii to its equivalent EBCDIC */

static const char asc2ebc[128] =
{
    0,  1,  2,  3, 55, 45, 46, 47, 22,  5, 37, 11, 12, 13, 14, 15,
   16, 17, 18, 19, 60, 61, 50, 38, 24, 25, 63, 39, 28, 29, 30, 31,
   64, 90,127,123, 91,108, 80,125, 77, 93, 92, 78,107, 96, 75, 97,
  240,241,242,243,244,245,246,247,248,249,122, 94, 76,126,110,111,
  124,193,194,195,196,197,198,199,200,201,209,210,211,212,213,214,
  215,216,217,226,227,228,229,230,231,232,233, 74,224, 79, 95,109,
  121,129,130,131,132,133,134,135,136,137,145,146,147,148,149,150,
  151,152,153,162,163,164,165,166,167,168,169,192,106,208,161,  7
} ;

/* This look up table converts ebcdic characters to 7 bit ascii characters.
   If no conversion exists then the value is 32.                            */

static const char ebc2asc[256] =
{
    0,  1,  2,  3, 32,  9, 32,127, 32, 32, 32, 11, 12, 13, 14, 15,
   16, 17, 18, 19, 32, 32,  8, 32, 24, 25, 32, 32, 28, 29, 30, 31,
   32, 32, 32, 32, 32, 10, 23, 27, 32, 32, 32, 32, 32,  5,  6,  7,
   32, 32, 22, 32, 32, 32, 32,  4, 32, 32, 32, 32, 20, 21, 32, 26,
   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 91, 46, 60, 40, 43, 93,
   38, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 36, 42, 41, 59, 94,
   45, 47, 32, 32, 32, 32, 32, 32, 32, 32,124, 44, 37, 95, 62, 63,
   32, 32, 32, 32, 32, 32, 32, 32, 32, 96, 58, 35, 64, 39, 61, 34,
   32, 97, 98, 99,100,101,102,103,104,105, 32, 32, 32, 32, 32, 32,
   32,106,107,108,109,110,111,112,113,114, 32, 32, 32, 32, 32, 32,
   32,126,115,116,117,118,119,120,121,122, 32, 32, 32, 32, 32, 32,
   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
  123, 65, 66, 67, 68, 69, 70, 71, 72, 73, 32, 32, 32, 32, 32, 32,
  125, 74, 75, 76, 77, 78, 79, 80, 81, 82, 32, 32, 32, 32, 32, 32,
   92, 32, 83, 84, 85, 86, 87, 88, 89, 90, 32, 32, 32, 32, 32, 32,
   48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 32, 32, 32, 32, 32, 32,
} ;

/** GLOBAL PROCEDURE *******************************************************/

void ebcdic_to_ascii( src, dst, n)
char *src;
char *dst;
int n;
  /* This converts the ascii format source character to an EBCDIC format
  destination character. */
{
  int i ;
  for (i=0;i<n;i++)
    dst[i] = ebc2asc[(unsigned char)src[i]] ;
} ;

/** GLOBAL PROCEDURE *******************************************************/

void ascii_to_ebcdic(src, dst, n)
char *src;
char *dst;
 int n;
  /* This converts the EBCDIC source to an ASCII destination. Characters
  which cannot be converted are converted to spaces. */
{
  int i ;
  for (i=0;i<n;i++)
    dst[i]=asc2ebc[(src[i] & 127)] ;
}

/** GLOBAL PROCEDURE ******************************************************/

void byte_swop_2 ( src, dst, n)
char *src;
char *dst;
int n;
  /* This flips the two bytes in a 2 byte integer pointed to by src and returns
  two bytes in dst */
{
  int i ;
  char tmp[2] ;            /* This is needed to prevent aliasing */

  for (i=0;i<(2*n);i+=2)
  {
    tmp[0]   = src[i] ;  tmp[1] = src[i+1] ;
    dst[i+1] = tmp[0] ;  dst[i] = tmp[1] ;
  } ;
} ;

/** GLOBAL PROCEDURE ******************************************************/

void byte_swop_4 (src, dst,  n)
char *src; 
char *dst;
int n;
  /* This is as above but for a four byte integer */
{
  int i ;
  char tmp[4] ;

  for (i=0;i<(4*n);i+=4)
  {
    tmp[0]   = src[i+3] ;  tmp[1]   = src[i+2] ;
    tmp[2]   = src[i+1] ;  tmp[3]   = src[i] ;
    dst[i]   = tmp[0] ;    dst[i+1] = tmp[1] ;
    dst[i+2] = tmp[2] ;    dst[i+3] = tmp[3] ;
  } ;
} ;

/** GLOBAL PROCEDURE ******************************************************/

void byte_swop_8 (src,  dst,  n)
char *src;
char *dst;
int n;
  /* This is as above but for a 8 byte integer */
{
  int i ;
  char tmp[8] ;

  for (i=0;i<(n*8);i+=8)
  {
    tmp[0]   = src[i+7] ;  tmp[1]   = src[i+6] ;
    tmp[2]   = src[i+5] ;  tmp[3]   = src[i+4] ;
    tmp[4]   = src[i+3] ;  tmp[5]   = src[i+2] ;
    tmp[6]   = src[i+1] ;  tmp[7]   = src[i] ;
    dst[i]   = tmp[0] ;    dst[i+1] = tmp[1] ;
    dst[i+2] = tmp[2] ;    dst[i+3] = tmp[3] ;
    dst[i+4] = tmp[4] ;    dst[i+5] = tmp[5] ;
    dst[i+6] = tmp[6] ;    dst[i+7] = tmp[7] ;
  } ;
} ;

/** GLOBAL PROCEDURE *******************************************************/

void short_to_long (src, dst, n)
char *src;
char *dst;
int n;
  /* This converts a little-endian 2 byte integer to a little-endian 4 byte
  integer. The integer is assumed to be signed.*/
{
  int i ;
  for (i=2*(n-1);i>=0;i-=2)
  {
    dst[i<<1]     = src[i] ;
    dst[(i<<1)+1] = src[i+1] ;
    dst[(i<<1)+2] = ( dst[(i<<1)+3] = (((int)src[i+1]<0) ? -1 : 0) ) ;
  } ;
} ;

/** GLOBAL PROCEDURE *********************************************************/

void short_to_long_uns (src, dst, n)
char *src;
char *dst;
int n;
  /* This is as above but assumes an unsigned integer */
{
  int i ;
  for (i=2*(n-1);i>=0;i-=2)
  {
    dst[i<<1]     = src[i] ;
    dst[(i<<1)+1] = src[i+1] ;
    dst[(i<<1)+2] = (dst[(i<<1)+3]=0) ;
  } ;
} ;

/** GLOBAL PROCEDURE *********************************************************/

void long_to_short (src, dst, n)
char *src;
char *dst;
int n;
  /* This converts a little endian signed integer of 4 bytes to one of two. If
  the resulting integer cannot be fitted in 2 bytes the value is forced to
  -32768 or +32767 according to the sign of the original integer. */
{
  int i ;

  for (i=0;i<(4*n);i+=4)
  {
    if (  ((src[i+2]==src[i+3]) && (src[i+3]==(char)-1))
       || ((src[i+2]==src[i+3]) && (src[i+3]==0) && (src[i+1]>0))  )
                        /* Highest bytes must be same */
                        /* Either 0,0 or -1,-1        */
                        /* or positive and <32767     */
    {
      dst[(i>>1)]   = src[i] ;                   /* OK .. do conversion.      */
      dst[(i>>1)+1] = src[i+1] ;
    }
    else
    {
      if ((int)src[i+3]<0)
      {
        dst[(i>>1)] = -1 ; dst[(i>>1)+1] = -1 ;  /* Negitive so make -32768   */
      }
      else
      {
        dst[(i>>1)] = -1 ; dst[(i>>1)+1] = 127 ; /* Positive so make +32767   */
      } ;
    } ;
  } ;
} ;

/** GLOBAL PROCEDURE ********************************************************/

void long_to_short_uns (src, dst, n)
char *src;
char *dst;
int n;
  /* As above but for unsigned integers. */
{
  int i ;

  for (i=0;i<(4*n);i+=4)
  {
    if ((src[i+2] | src[i+3])!=0)
    {
      dst[(i>>1)] = -1 ; dst[(i>>1)+1] = -1 ;    /* Force to 65535 */
    }
    else
    {
      dst[(i>>1)] = src[i] ; dst[(i>>1)+1] = src[i+1] ;
    } ;
  } ;
} ;

/** GLOBAL PROCEDURE **********************************************************/

void short_s_to_uns (src, dst, n)
char *src;
char *dst;
int n;
  /* Converts signed little endian short to unsigned little endian short */
{
  int i ;

  for (i=0;i<(2*n);i+=2)
  {
    dst[i]   = ((int)src[i+1]<0) ? 0 : src[i] ;
    dst[i+1] = ((int)src[i+1]<0) ? 0 : src[i+1] ;
  } ;
} ;

/** GLOBAL PROCEDURE **********************************************************/

void short_uns_to_s (src, dst, n)
char *src;
char *dst;
int n;

  /* Converse of above */
{
  int i ;

  for (i=0;i<(2*n);i+=2)
  {
    dst[i]   = ((int)src[i+1]<0) ?  -1 : src[i] ;
    dst[i+1] = ((int)src[i+1]<0) ? 127 : src[i+1] ;
  } ;
} ;

/** GLOBAL PROCEDURE **********************************************************/

void long_s_to_uns (src, dst, n)
char *src;
char *dst;
int n;
  /* Converts signed little endian longs to unsigned little endian longs */
{
  int i ;

  for (i=0;i<(4*n);i+=4)
  {
    if ((int)src[i+3]<0)
    {
      dst[i]=(dst[i+1]=(dst[i+2]=(dst[i+3]=0))) ;
    }
    else
    {
      dst[i]  =src[i]   ; dst[i+1]=src[i+1] ;
      dst[i+2]=src[i+2] ; dst[i+3]=src[i+3] ;
    } ;
  } ;
} ;

/** GLOBAL PROCEDURE **********************************************************/

void long_uns_to_s (src, dst, n)
char *src;
char *dst;
int n;
  /* Converse of above. */
{
  int i ;

  for (i=0;i<(4*n);i+=4)
  {
    if ((int)src[i+3]<0)
    {
      dst[i]  =(dst[i+1]=(dst[i+2]= -1)) ;
      dst[i+3]=127 ;
    }
    else
    {
      dst[i]  =src[i]   ; dst[i+1]=src[i+1] ;
      dst[i+2]=src[i+2] ; dst[i+3]=src[i+3] ;
    } ;
  } ;
} ;

/** GLOBAL PROCEDURE **********************************************************/

void nil_conv (src,  dst,  sz,  n)
char *src;
char *dst;
int sz;
int n;
  /* Transfers "sz*n" bytes from the source buffer to the destination buffer */
{
  unsigned int i;

  for (i=0;i<(unsigned int)(n*sz);i++)
    dst[i]=src[i] ;
} ;

/** LOCAL PROCEDURE ***********************************************************/

static void long_to_real32 (src, dst , exponent,  s,  n)
char *src;
char *dst;
int exponent;
 int s;
 int n;
  /* This converts a signed long integer to a 32bit real. If the integer
  "s" is non zero then the source is assumed to be a signed integer.
  Otherwise it is assumed to be unsigned. */
{
  unsigned long mant,
                test ;
  unsigned long sign,
                exp ;
  int           i,j ;

  for (j=0;j<(4*n);j+=4)
  {
    if ((src[j]|src[j+1]|src[j+2]|src[j+3])==0)
    {
      dst[j]=(dst[j+1]=(dst[j+2]=(dst[j+3]=0))) ;
      /* Representation of zero in 32 bit floating point. */
    }
    else
    {
      mant = 0 ;
      for (i=3;i>=0;i--)                        /* Converts source string to */
      {                                         /* long integer.             */
        mant <<= 8 ;
        mant = mant + (unsigned char) src[j+i] ;
      } ;
      if (((int)src[j+3]<0) && (s))
      {
        sign = 256 ;
        mant = (~mant)+1 ;
      }
      else
        sign = 0 ;

      test = 2 ;                        /* This loop moves a mask value over the */
      exp  = 0 ;                        /* mantissa until the most significant   */
      while ((test<=mant) && (test!=0)) /* set digit of the mask moves past the  */
      {                                 /* most significant set dig of the num.  */
        test <<= 1 ;
        exp  ++  ;                      /* MASK 00000000000100000000000000000000 */
      } ;                               /* mant 00000000000010010010111101000111 */

      if (test!=0)
        test >>= 1 ;
      else
        test = ( (unsigned long)1 << 31 ) ;
      mant = (mant^test) ;               /* Now eliminate most significant digit. */

      if (exp<23)                  /* Here we are extracting 23 digits from the */
      {                            /* mantissa. To do this it is necessary to   */
        mant<<=(23-exp);           /* shift either left or right according to   */
      }                            /* the original position of the most signif- */
      else                         /* icant digit. At the end, the lowest 23    */
      {                            /* bits of the variable "mant" contain the   */
        mant>>=(exp-23) ;          /* mantissa for incorporating in the real.   */
      } ;

      /* Now combine mantissa, exponent and sign to get fp real */
      mant = mant | (((exp+exponent)|sign)<<23) ;

      /* Now convert unsigned long to destination buffer */
      for (i=0;i<=3;i++)
      {
        dst[i+j] = (unsigned char) (mant&255) ;
        mant >>= 8 ;
      } ;
    } ;
  } ;
} ;

/** LOCAL PROCEDURE ***********************************************************/

static void long_to_real64(src,  dst, exponent,  s,  n)
char *src;
char *dst;
int exponent;
int s;
 int n;
  /* This converts a signed long integer to a 64bit real. If the integer
  "s" is non zero then the source is assumed to be a signed integer.
  Otherwise it is assumed to be unsigned. */
{
  unsigned long mant,
                test ;
  unsigned long sign,
                exp ;
  int           i,j ;

  for (j=4*(n-1);j>=0;j-=4)
  {
    if ((src[j]|src[j+1]|src[j+2]|src[j+3])==0)
    {
      dst[(j<<1)]=(dst[(j<<1)+1]=(dst[(j<<1)+2]=(dst[(j<<1)+3]=0))) ;
      dst[(j<<1)+4]=(dst[(j<<1)+5]=(dst[(j<<1)+6]=(dst[(j<<1)+7]=0))) ;
      /* Representation of zero in 64 bit floating point. */
    }
    else
    {
      mant = 0 ;
      for (i=3;i>=0;i--)                        /* Converts source string to */
      {                                         /* long integer.             */
        mant <<= 8 ;
        mant = mant + (unsigned char) src[i+j] ;
      } ;
      if (((int)src[j+3]<0) && (s))
      {
        sign = 2048 ;
        mant = (~mant)+1 ;
      }
      else
        sign=0 ;

      test = 2 ;                       /* This loop moves a mask value over the */
      exp  = 0 ;                       /* mantissa until the most significant   */
      while ((test<=mant) && (test!=0)) /* set digit of the mask moves past the  */
      {                                /* most significant set dig of the num.  */
        test <<= 1 ;
        exp  ++  ;                     /* MASK 00000000000100000000000000000000 */
      } ;                              /* mant 00000000000010010010111101000111 */

      if (test!=0)
        test >>= 1 ;
      else
        test = ( (unsigned long) 1 << 31 ) ;
      mant = (mant^test) ;               /* Now eliminate most significant digit. */

      mant <<= (32-exp) ;              /* Move mantissa so first binary place   */
                                       /* is at bit 31 of the long int.         */

      /* Now convert sign,exponent and mantissa to destination buffer */
      dst[(j<<1)]=0 ; dst[(j<<1)+1]=0 ;
      dst[(j<<1)+2]= (((unsigned char) (mant&15))<<4) ;
      mant >>= 4 ;
      for (i=3;i<=5;i++)
      {
        dst[(j<<1)+i] = (unsigned char) (mant&255) ;
        mant >>= 8 ;
      } ;
      mant=(((exp+exponent)|sign)<<4)|mant ;
      for (i=6;i<=7;i++)
      {
        dst[(j<<1)+i] = (unsigned char) (mant&255) ;
        mant >>= 8 ;
      } ;
    } ;
  } ;
} ;

/** LOCAL PROCEDURE ***********************************************************/

static void long_to_ibm32( src, dst, s,  n)
char *src;
char *dst;
int s;
int n;
  /* Convert a long (signed if s=1) to an IBM 32 bit real format number. */
{
  unsigned long mant,
                test ;
  unsigned long sign,
                exp ;
  int           i,j ;

  for (j=0;j<(4*n);j+=4)
  {
    if ((src[j]|src[j+1]|src[j+2]|src[j+3])==0)
    {
      dst[j]=(dst[j+1]=(dst[j+2]=(dst[j+3]=0))) ;
      /* Representation of zero in 32 bit floating point. */
    }
    else
    {
      mant = 0 ;
      for (i=3;i>=0;i--)                        /* Converts source string to */
      {                                         /* long integer.             */
        mant <<= 8 ;
        mant = mant + (unsigned char) src[j+i] ;
      } ;
      if (((int)src[j+3]<0) && (s))
      {
        sign = 128 ;
        mant = (~mant)+1 ;
      }
      else
        sign = 0 ;

      test = 2 ;                        /* This loop moves a mask value over the */
      exp  = 0 ;                        /* mantissa until the most significant   */
      while ((test<=mant) && (test!=0)) /* set digit of the mask moves past the  */
      {                                 /* most significant set dig of the num.  */
        test <<= 1 ;
        exp  ++  ;                      /* MASK 00000000000100000000000000000000 */
      } ;                               /* mant 00000000000010010010111101000111 */

      exp = (exp >> 2) ;

      /* At this point "exp" is the nyble number of the most significant nyble */
      /* which contains the most significant bit. It is then necessary to shift*/
      /* the mantissa by multiples of four bits to shift the nyble to the 6th  */
      /* nyble possition. */

      if (exp<5)
        mant<<=(4*(5-exp)) ;          /* Shift nybles up */
      else
        mant>>=(4*(exp-5)) ;          /* Shift down */

      /* Now combine mantissa, exponent and sign to get fp real */
      mant = mant | (((exp+ibm_exp_32+1)|sign)<<24) ;

      /* Now convert unsigned long to destination buffer */
      for (i=0;i<=3;i++)
      {
        dst[i+j] = (unsigned char) (mant&255) ;
        mant >>= 8 ;
      } ;
    } ;
  } ;
} ;

/** GLOBAL PROCEDURES *********************************************************/

void long_uns_to_ieee32(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts an unsigned 4 byte integer to a 32 bit ieee floating point real.*/
{
  long_to_real32(src,dst,ieee_exp_32,0,n) ;
} ;

void long_s_to_ieee32( src, dst,  n)
char *src;
char *dst;
int n;
  /* Converts a signed 4 byte integer to a 32 bit ieee floating point real. */
{
  long_to_real32(src,dst,ieee_exp_32,1,n) ;
} ;

void long_uns_to_ieee64(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts an unsigned 4 byte integer to a 64 bit ieee floating point real.*/
{
  long_to_real64(src,dst,ieee_exp_64,0,n) ;
} ;

void long_s_to_ieee64(src, dst, n)
char *src;
char *dst;
int n;

  /* Converts a signed 4 byte integer to a 64 bit ieee floating point real. */
{
  long_to_real64(src,dst,ieee_exp_64,1,n) ;
} ;

void long_uns_to_convex32(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts an unsigned 4 byte integer to a 32 bit convex floating point real.*/
{
  long_to_real32(src,dst,convex_exp_32,0,n) ;
} ;

void long_s_to_convex32( src, dst,  n)
char *src;
char *dst;
int n;
  /* Converts a signed 4 byte integer to a 32 bit convex floating point real. */
{
  long_to_real32(src,dst,convex_exp_32,1,n) ;
} ;

void long_uns_to_convex64(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts an unsigned 4 byte integer to a 64 bit convex floating point real.*/
{
  long_to_real64(src,dst,convex_exp_64,0,n) ;
} ;

void long_s_to_convex64(src, dst, n)
char *src;
char *dst;
int n;

  /* Converts a signed 4 byte integer to a 64 bit convex floating point real. */
{
  long_to_real64(src,dst,convex_exp_64,1,n) ;
} ;

void long_uns_to_ibm32( src, dst,  n)
char *src;
char *dst;
int n;
  /* Converts an unsigned 4 byte integer to a 32 bit IBM floating point real. */
{
  long_to_ibm32(src,dst,0,n) ;
} ;

void long_s_to_ibm32( src, dst,  n)
char *src;
char *dst;
int n;
  /* Converts a signed 4 byte integer to a 32 bit IBM floating point real. */
{
  long_to_ibm32(src,dst,1,n) ;
} ;

void long_uns_to_ibm64(src, dst, n)
char *src;
char *dst;
int n;
  /* Does bugger all */
{
  abort() ;
} ;

void long_s_to_ibm64(src, dst, n)
char *src;
char *dst;
int n;
  /* Again Bugger All */
{
  abort() ;
} ;


/** GLOBAL PROCEDURE **********************************************************/

void single_to_double( src, dst,  n)
char *src;
char *dst;
int n;
  /* This procedure converts a 32 bit fp number to a 64 bit fp number in a
  given format. If a number cannot be converted it is returned as zero. */
{
  long              exp  ;
  unsigned long     mant,
                    sign ;
  int  i,j ;

  for (j=4*(n-1);j>=0;j-=4)
  {
    for (i=3;i>=0;i--)                        /* Get source into long int. */
    {
      mant <<= 8 ;
      mant = mant + (unsigned char) src[i+j] ;
    } ;

    sign = ( mant & 0x80000000 ) >> 20 ;
    exp  = ( mant & 0x7f800000 ) >> 23 ;
    mant = ( mant & 0x007fffff ) <<  9 ;

    switch ((int)exp)
    {
      case   0 : break ;
      case 255 : exp=2047 ;
                 break ;
      default  : exp += 896 ;
    } ;

    /* Now convert sign,exponent and mantissa to destination buffer */
    dst[(j<<1)]  = 0 ; dst[(j<<1)+1]=0 ;
    dst[(j<<1)+2]= (((unsigned char) (mant&0x0f))<<4) ;
    mant >>= 4 ;
    for (i=3;i<=5;i++)
    {
      dst[(j<<1)+i] = (unsigned char) (mant&0xff) ;
      mant >>= 8 ;
    } ;
    mant=(((exp)|sign)<<4)|mant ;
    for (i=6;i<=7;i++)
    {
      dst[(j<<1)+i] = (unsigned char) (mant&0xff) ;
      mant >>= 8 ;
    } ;
  } ;
} ;

/** GLOBAL PROCEDURE *********************************************************/

void double_to_single(src, dst, n)
char *src;
char *dst;
int n;
  /* This procedure converts a 64 bit real to a 32 bit real. If the conversion
  cannot be done the return value is zero. */
{
  long          exp ;
  unsigned long mant,
                sign ;
  int  i,j ;

  for (j=0;j<(8*n);j+=8)
  {
    mant = 0 ;
    for (i=6;i>=3;i--)                        /* Get mantissa into long int. */
    {
      mant <<= 8 ;
      mant = mant+((((unsigned char) src[j+i])&0x0f)<<4)
                 +((((unsigned char) src[j+i-1])&0xf0)>>4) ;
    } ;
    mant>>=9 ;
    exp  = ((((unsigned char) src[j+7])&0x7f)<<4)
           + ((((unsigned char) src[j+6])&0xf0)>>4) ;
    sign = ((int)src[j+7]<0) ? 256 : 0 ;

    switch ((int)exp)
    {
      case    0 : break ;
      case 2047 : exp = 255 ;
                  break ;
      default   : exp -= 896 ;
                  if ((exp<1) || (exp>254))
                  {
                    exp=0 ; mant=0 ; sign=0 ;
                  } ;
    } ;

    /* Now combine mantissa, exponent and sign to get fp real */
    mant = mant | (((exp)|sign)<<23) ;

    /* Now convert unsigned long to destination buffer */
    for (i=0;i<=3;i++)
    {
      dst[(j>>1)+i] = (unsigned char) (mant&0xff) ;
      mant >>= 8 ;
    } ;
  } ;
} ;

/** MORE GLOBAL PROCEDURES ***************************************************/

void single_to_double_ibm(src, dst, n)
char *src;
char *dst;
int n;
  /* Does bugger all */
{
  abort() ;
} ;

void double_to_single_ibm(src, dst, n)
char *src;
char *dst;
int n;
  /* Does bugger all */
{
  abort() ;
} ;

/** LOCAL PROCEDURE **********************************************************/

static void exp_shift_32( src, dst,  shift,  n)
char *src;
char *dst;
int shift;
int n;
  /* This procedure adds to the exponent of a 32 bit real the shift vector
  "shift". If this takes the real into any reserved keyword 0 is returned. */
{
  unsigned long exp=0 ;
  int j ;

  for (j=0;j<(4*n);j+=4)
  {
    exp  = ((((unsigned char) src[j+3])&0x7f)<<1)
          + ((((unsigned char) src[j+2])&0x80)>>7) ;

    switch ((int)exp)
    {
      case   0 :
      case 255 : dst[j]  =src[j]   ; dst[j+1]=src[j+1] ;
                 dst[j+2]=src[j+2] ; dst[j+3]=src[j+3] ;
                 break ;
      default  : exp=exp+shift ;
                 if ((exp<1) || (exp>254))
                 {
                   dst[j]=(dst[j+1]=(dst[j+2]=(dst[j+3]=0))) ;
                 }
                 else
                 {
                   dst[j]  =src[j]   ; dst[j+1]=src[j+1] ;
                   dst[j+2]=(src[j+2]&0x7f) | ((unsigned char)((exp&0x01)<<7)) ;
                   dst[j+3]=(src[j+3]&0x80) | ((unsigned char)((exp&0xfe)>>1)) ;
                 } ;
    } ;
  } ;
} ;

/** LOCAL PROCEDURE ***********************************************************/

static void exp_shift_64(src,  dst,  shift,  n)
char *src;
char *dst;
int shift;
int n;
  /* As above but for 64 bit floating point numbers. */
{
  unsigned long exp=0 ;
  int j ;

  for (j=0;j<(8*n);j+=8)
  {
    exp  = ((((unsigned char) src[j+7])&0x7f)<<4) +
            ((((unsigned char) src[j+6])&0xf0)>>4) ;

    switch ((int)exp)
    {
      case    0 :
      case 2047 : dst[j]  =src[j] ;   dst[j+1]=src[j+1] ;
                  dst[j+2]=src[j+2] ; dst[j+3]=src[j+3] ;
                  dst[j+4]=src[j+4] ; dst[j+5]=src[j+5] ;
                  dst[j+6]=src[j+6] ; dst[j+7]=src[j+7] ;
                  break ;
      default   : exp=exp+shift ;
                  if ((exp<1) || (exp>254))
                  {
                    dst[j]  =(dst[j+1]=(dst[j+2]=(dst[j+3]=0))) ;
                    dst[j+4]=(dst[j+5]=(dst[j+6]=(dst[j+7]=0))) ;
                  }
                  else
                  {
                    dst[j]  =src[j] ;   dst[j+1]=src[j+1] ;
                    dst[j+2]=src[j+2] ; dst[j+3]=src[j+3] ;
                    dst[j+4]=src[j+4] ; dst[j+5]=src[j+5] ;
                    dst[j+6]=(src[j+6]&0x0f) | ((unsigned char)((exp&0x00f)<<4)) ;
                    dst[j+7]=(src[j+7]&0x80) | ((unsigned char)((exp&0x7f0)>>4)) ;
                  } ;
    } ;
  } ;
} ;

/** GLOBAL PROCEDURES ********************************************************/

void ieee_to_convex_32(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts IEEE 32 bit real to Convex 32 bit real. */
{
  exp_shift_32(src,dst,(convex_exp_32-ieee_exp_32),n) ;
} ;

void ieee_to_convex_64(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts IEEE 64 bit real to Convex 64 bit real. */
{
  exp_shift_64(src,dst,(convex_exp_64-ieee_exp_64),n) ;
} ;

void convex_to_ieee_32(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts Convex 32 bit real to IEEE 32 bit real. */
{
  exp_shift_32(src,dst,(ieee_exp_32-convex_exp_32),n) ;
} ;

void convex_to_ieee_64(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts Convex 64 bit real to IEEE 64 bit real. */
{
  exp_shift_64(src,dst,(ieee_exp_64-convex_exp_64),n) ;
} ;

/** LOCAL PROCEDURE ************************************************************/

static void ibm_to_real_32( src, dst, exponent, n)
char *src;
char *dst;
int exponent;
int n;
  /* This converts an ibm real into an other type of real, depending on the size
  of the exponent passed as the arguement. */
{
  unsigned long mant,test,sign ;
  long          exp ;
  int           i,j ;

  for (i=0;i<(4*n);i+=4)
  {
    if ((src[i]|src[i+1]|src[i+2])==0)    /* ibm representation of 0 */
    {
      dst[i] = 0 ; dst[i+1] = 0 ; dst[i+2] = 0 ; dst[i+3] = 0 ;
    }
    else
    {
      mant = 0 ;                  /* First extract mantissa,exponent & sign */
      for (j=2;j>=0;j--)
      {
        mant<<=8 ;
        mant = mant + (unsigned long) (((unsigned char)src[i+j])&255) ;
      } ;
      /* Multiply exponent by 4 to convert from powers of 16 to powers of 2 */
      exp  = (((long)(((unsigned char)src[i+3])&127))-ibm_exp_32)*4 ;
      sign = ((unsigned long)(((unsigned char)src[i+3])&128)) ;

      test = 2 ;                        /* This loop moves a mask value over the */
      j    = 0 ;                        /* mantissa until the most significant   */
      while ((test<=mant) && (test!=0)) /* set digit of the mask moves past the  */
      {                                 /* most significant set dig of the num.  */
        test <<= 1 ;
        j ++ ;                          /* MASK 00000000000100000000000000000000 */
      } ;                               /* mant 00000000000010010010111101000111 */

      /* j is bit number of most significant bit. */

      if (test==0)                      /* Shift back to overlay most significant*/
        test=((unsigned long)1)<<31 ;   /* digit. */
      else
        test>>=1 ;

      mant = mant^test ;                /* Kill most significant digit. */

      if (j<23)                         /* Now shift mantissa to normalise */
        mant<<=(23-j) ;                 /* the floating point. */
      else
        mant>>=(j-23) ;

      exp = exp+j-24+exponent ;         /* Convert exponent */

      mant = (mant&0x007fffff)|(sign<<24)|((unsigned long)(exp&0xff)<<23) ;

      for (j=0;j<4;j++)                 /* Write output */
      {
        dst[i+j] = (unsigned char) (mant&0xff) ;
        mant>>=8 ;
      } ;
    } ;
  } ;
} ;

/** LOCAL PROCEDURE ***********************************************************/

static void real_to_ibm_32 ( src, dst, exponent, n)
char *src;
char *dst;
int exponent;
int n;
  /* This procedure converts a convex or ieee real to an ibm format floating
  point number */
{
  long              exp  ;
  unsigned long     mant,
                    sign ;
  int  i,j ;

  for (i=0;i<(4*n);i+=4)
  {
    mant = 0 ;
    for (j=3;j>=0;j--)                        /* Get source into long int. */
    {
      mant <<= 8 ;
      mant = mant + (unsigned char) src[i+j] ;
    } ;

    sign = ( mant & 0x80000000 ) ;
    exp  = (( mant & 0x7f800000 ) >> 23) ;
    mant = ( mant & 0x007fffff ) ;

    switch ((int)exp)
    {
      case   0 :                               /* Zero */
      case 255 : mant=0 ; exp=0 ; sign=0 ;     /* Illegal number */
                 break ;
    } ;

    if ((mant|exp|sign)==0)
    {
      dst[i] = 0 ; dst[i+1] = 0 ; dst[i+2] = 0 ; dst[i+3] = 0 ;
    }
    else
    {
      exp = exp - exponent ;
      mant = mant | 0x00800000 ;        /* Re-add phantom 1 */
      mant = mant>>(3-(exp&0x03)) ;     /* Shift mantissa for powers of 16 */
      exp  = (exp>>2)+1+ibm_exp_32 ;    /* Convert exponent to powers of 16 */

      mant = (mant&0x00ffffff)|sign|((((unsigned long)exp)&0x0000007f)<<24) ;

      for (j=0;j<4;j++)                 /* Write output */
      {
        dst[i+j] = (unsigned char) (mant&0xff) ;
        mant>>=8 ;
      } ;
    } ;
  } ;
} ;

/** GLOBAL PROCEDURES *********************************************************/

void ibm_to_convex_32(src, dst, n)
char *src;
char *dst;
int n;

  /* Converts ibm to convex style floating point numbers */
{
  ibm_to_real_32(src,dst,convex_exp_32,n) ;
} ;

void ibm_to_ieee_32(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts ibm to ieee style floating point numbers */
{
  ibm_to_real_32(src,dst,ieee_exp_32,n) ;
} ;

void convex_to_ibm_32(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts convex to ibm style floating point numbers */
{
  real_to_ibm_32(src,dst,convex_exp_32,n) ;
} ;

void ieee_to_ibm_32(src, dst, n)
char *src;
char *dst;
int n;
  /* Converts ieee to ibm style floating point numbers */
{
  real_to_ibm_32(src,dst,ieee_exp_32,n) ;
} ;

void ibm_to_convex_64(src, dst, n)
char *src;
char *dst;
int n;
  /* Does bugger all */
{
  abort() ;
} ;

void ibm_to_ieee_64(src, dst, n)
char *src;
char *dst;
int n;
  /* Does bugger all */
{
  abort() ;
} ;

void convex_to_ibm_64(src, dst, n)
char *src;
char *dst;
int n;
  /* Does bugger all */
{
  abort() ;
} ;

void ieee_to_ibm_64(src, dst, n)
char *src;
char *dst;
int n;
  /* Does bugger all */
{
  abort() ;
} ;
