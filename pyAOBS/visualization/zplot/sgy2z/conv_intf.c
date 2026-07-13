/* This file provdes an interface between Fortran and C so that the
   functions in $MARINE_S/lib/ffc/conv.c may be called by either.
   This method - adding an underscore to the function name - is preferred to
   the use of the # pragma directive as it avoids implementation 
   dependent code creeping into the main Fortran programs.          /*

/* Similar functions can be added for all the conversion functions in conv.c */


void ieee_to_ibm_32_(src,dst, n)

/* Convert an ieee real to an ibm real  - note that the number must be
   byte swapped before and after the conversion fuinction is called     */

char *src;
char *dst;
int *n;
  /* Converts ibm to ieee style floating point numbers */
{

  byte_swop_4(src,dst,*n) ;
  ieee_to_ibm_32(dst,dst,*n) ;
  byte_swop_4(dst,dst,*n) ;
}
void ibm_to_ieee_32_(src,dst, n)

/* Convert an ibm real to an ieee real  - note that the number must be
   byte swapped before and after the conversion fuinction is called     */

char *src;
char *dst;
int *n;
  /* Converts ibm to ieee style floating point numbers */
{

  byte_swop_4(src,dst,*n) ;
  ibm_to_ieee_32(dst,dst,*n) ;
  byte_swop_4(dst,dst,*n) ;
}
 
void long_s_to_ibm32_(src,dst, n)

/* Convert a signed 4 byte integer to an ibm real  - note that the number 
 must be byte swapped before and after the conversion function is called     */

char *src;
char *dst;
int *n;
{
  byte_swop_4(src,dst,*n) ;
  long_s_to_ibm32(dst,dst,*n) ;
  byte_swop_4(dst,dst,*n) ;
}

