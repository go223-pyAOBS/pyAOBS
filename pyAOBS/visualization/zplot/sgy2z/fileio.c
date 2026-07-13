/******************************************************************************/
/*----------------------------------------------------------------------------*/
/* Author:         Clare Enright - fns from Andrew Scholan                    */ /*============================================================================*/
/* Filename:       fileio.c                                                   */
/* Compiler:       Convex C Compiler                                          */
/*----------------------------------------------------------------------------*/
/* SYNOPSIS:      Opens/closes files of given name for read/write             */
/*                Reads or writes a block of data                             */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#define  max_file  21
#define FILENAME_MAX 256

struct file_str { FILE *f;                  /* C file variable   */
                  int  isopen;              /* 0=closed, 1=open  */
                  int  isinput; } ;         /* 0=output, 1=input */

struct file_str files[max_file] ;

int open_input ();

/** OPEN & CLOSE **************************************************************/

open_input_(handle_ptr)
int  *handle_ptr ;        /* channel number */



/*   printf("max file name length: %d", FILENAME_MAX);  */
/*   printf ( "Try to open  %s for input ; length %d",
                            name_ptr,namelength );    */
{

  char name_ptr[FILENAME_MAX] ;          /* file name */
  printf( "File name for input: " );
  gets( name_ptr);


  if (*handle_ptr<max_file)
  {
    if (!files[*handle_ptr].isopen)
    {
      files[*handle_ptr].f=fopen(name_ptr,"rb") ;
      if (files[*handle_ptr].f != NULL)
      {
        files[*handle_ptr].isopen  = 1 ;
        files[*handle_ptr].isinput = 1 ;
        printf ( "open succeeded " );
      }
      else
      {
      printf( "File open failed" );
       };
    } ;
  } ;
  return 0 ;
} ;

open_output_(handle_ptr)
int  *handle_ptr ;        /* channel number */

{
char name_ptr[FILENAME_MAX] ;         /* file name */


  printf( "File name for output: " );
  gets( name_ptr);


  printf ( "Try to open  %s for output ", name_ptr );
  if (*handle_ptr<max_file)
  {
    if (!files[*handle_ptr].isopen)
    {
      files[*handle_ptr].f=fopen(name_ptr,"wb") ;
      if (files[*handle_ptr].f != NULL)
      {
        files[*handle_ptr].isopen  = 1 ;
        files[*handle_ptr].isinput = 0 ;
        printf ( "open succeeded" );
      }
      else
      {
      printf( "File open failed" );
       };

    } ;
  } ;
  return 0 ;
} ;

close_input_(handle_ptr)
int  *handle_ptr ;       /* channel number */
{
  if (*handle_ptr<max_file)
  {
    if (files[*handle_ptr].isopen)
    {
      fclose(files[*handle_ptr].f) ;
      files[*handle_ptr].isopen = 0 ;
    }
    else
      {
      printf( "File close failed" );
       } ;
  } ;
  return 0 ;
} ;

close_output_(handle_ptr)
int  *handle_ptr ;      /* channel number */
{
  if (*handle_ptr<max_file)
  {
    if (files[*handle_ptr].isopen)
    {
      fclose(files[*handle_ptr].f) ;
      files[*handle_ptr].isopen = 0 ;
    }
    else
      {
      printf( "File close failed" );
      };
  } ;
  return 0 ;
} ;


/** BLOCK READ AND WRITE ******************************************************/

block_read_(handle_ptr,address,size)
int  *handle_ptr ;     /* channel number */
char *address ;        /* pointer to space for data */
int  *size ;           /* number of bytes to read */
{
  if (*handle_ptr<max_file)
  {
    if ( (files[*handle_ptr].isopen) && (files[*handle_ptr].isinput) )
    {
      fread(address,1,*size,files[*handle_ptr].f) ;
    }
   else
      {
      printf( "Read failed: open $d, for input %d",
          files[*handle_ptr].isopen,   files[*handle_ptr].isinput );
       };
  } ;
  return 0 ;
} ;

block_write_(handle_ptr,address,size)
int  *handle_ptr ;    /* channel number */
char *address ;       /* pointer to space containing data */
int  *size ;           /* number of bytes to write */

{
  if (*handle_ptr<max_file)
  {
    if ( (files[*handle_ptr].isopen) && (!files[*handle_ptr].isinput) )
    {
      fwrite(address,1,*size,files[*handle_ptr].f) ;
      printf( "Write succeeded " );
    }
    else
    {
      printf( "Write failed: open $d, for input %d",
          files[*handle_ptr].isopen,   files[*handle_ptr].isinput );
    };
  } ;
  return 0 ;
} ;


/** End of file test **********************************************************/

int end_of_file_(handle_ptr)
int *handle_ptr ;   /* channel number */
{
  int eof=1 ;
  if (*handle_ptr<max_file)
  {
    if ( (files[*handle_ptr].isopen) && (files[*handle_ptr].isinput) )
    {
      eof=feof(files[*handle_ptr].f) ;
    } ;
  } ;
  return eof ;
} ;

