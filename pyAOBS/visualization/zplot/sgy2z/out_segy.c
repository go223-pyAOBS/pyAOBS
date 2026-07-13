/******************************************************************************/
/* All source code copyright Cambridge University (c) 1989,1990               */
/*----------------------------------------------------------------------------*/
/* Author:         Andrew Scholan                                             */
/*============================================================================*/
/* Filename:       binfile.c                                                  */
/* Compiler:       SunOS C Compiler (not ANSI standard)                       */
/*----------------------------------------------------------------------------*/
/* SYNOPSIS:                                                                  */
/*   Interface for fortran programs such that seg-y data can be output        */
/*    Note that the format is not true Seg-y as the inter-record gaps         */
/*    are incorrect. A tape written by these functions will not be readable   */
/*    by SKS. Copy tape using program ffcseg to get true seg-y tape           */
/******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#define  max_file  21


/* This look up table converts ebcdic characters to 7 bit ascii characters.
   If no conversion exists then the value is 32.                            */

static char ebc2asc[256] =
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



struct file_str { FILE *f;                  /* C file variable   */
                  int  isopen;              /* 0=closed, 1=open  */
                  int  isinput; } ;         /* 0=output, 1=input */

struct file_str files[max_file] ;

struct reel_hdr
    {
    char   header[3200];            /* Character header */
    long   job_id;                  /* Job id number    */
    long   line_no;                 /* Line number      */
    long   reel_no;                 /* Reel number      */
    short  n_trace;                 /* no of traces per record */
    short  n_auxtr;                 /* no of auxiliary traces */
    short  time_int;                /* sample interval (micro s ) */
    short  field_int;               /* interval in field record   */
    short  n_samp;                  /* no of samples per trace    */
    short  field_n_samp;            /* no of samples/tr in field  */
    short  format;                  /* format code                */
    short  cdp_fold;                /* no of data traces/ cdp ensemble */
    short  tr_sort_code;            /* trace sorting code */
    short  vert_sum_code;           /* vertical sum code    */
    short  sweep_fr_st;             /* sweep frequency at start */
    short  sweep_fr_end;            /* sweep frequency at end*/
    short  sweep_len;               /* sweep length (ms) */
    short  sweep_type;              /* sweep type code */
    short  sweep_tr_no;             /* trace no. of sweep ch. */
    short  sweep_tap_st;            /* sweep tr taper length at start (ms) */
    short  sweep_tap_end;           /* sweep tr taper length at end (ms) */
    short  sweep_tap_typ;           /* taper type code */
    short  correlated;              /* correlated data code */
    short  bin_gain;                /* binary gain recovered code */
    short  amp_code;                /* amplitude recovery method */
    short  meas_code;               /* measurement system */
    short  sign_pol;                /* impulse signal polarity */
    short  pol_code;                /*  vibratory polarity code */
    short  unused[170];             /* undefined fields */
    };

struct trace
    {
    long   trace_no;                /* Trace no within line    */
    long   trace_reel;              /* Trace no within reel      */
    long   field_rec;               /* Original field record no     */
    long   field_trace;             /* Trace no within field record */
    long   source_no;               /* Energy source point no */
    long   cdp_no;                  /* CDP ensemble no */
    long   trace_cdp;               /* Trae no within CDP */
    short  trace_id;                /* Trace identification code */
    short  vert_sum_no;             /* No of vertically summed traces  */
    short  hort_stack_no;           /* No of horizontally stacked traces  */
    short  data_use;                /* 1=production; 2=test */
    long   distance;                /* distance from source to receiver  */
    long   rcvr_elev;               /* receiver group elevation */
    long   src_elev;                /* source elevation */
    long   src_depth;               /* Source depth    */   
    long   rcvr_datum_elev;         /* datum elevation at receiver*/
    long   src_datum_elev;          /* datum elevation at source */
    long   src_water_depth;         /* water depth at source*/
    long   rcvr_water_depth;          /* water depth at receiver*/
    short  elev_scaler;             /* scaler for elevs. and depths  */
    short  co_ord_scaler;           /* scaler for co-ords */
    long   x_src;                   /* x co-ord at source */
    long   y_src;                   /* y co-ord at source */
    long   x_rcvr;                  /* x co-ord at receiver */
    long   y_rcvr;                  /* y co-ord at receiver */
    short  units;                   /* co-ordinate units */
    short  sed_vel;                 /* weathering velocity */
    short  base_vel;                /* sub_weathering velocity */
    short  time_src;                /* Uphole time at source */
    short  time_rcvr;               /* Uphole time at receiver */
    short  static_src;              /* Source static correction */
    short  static_rcvr;             /* Group static correction */
    short  static_tot;              /* Total static applied */
    short  lag_A;                   /* Lag time A */
    short  lag_B;                   /* Lag time B */
    short  delay;                   /* Delay recording time   */
    short  mute_start;              /* Mute time - start   */
    short  mute_end;                /* Mute time - end  */
    short  nsamps;                  /* No. of samples in this trace */
    short  intvl;                   /* Sample interval (microsecs)  */
    short  gain_type;               /* Gain type of field instruments  */
    short  gain_const;              /* Instrument gain constant      */
    short  gain_initial;            /* Instrument initial gain (dB)  */
    short  correlated;              /* Correlated ( 1 = yes, 2 = no )  */
    short  sweep_fr_start;          /* Sweep frequency at start     */
    short  sweep_fr_end;            /* Sweep frequency at end     */
    short  sweep_len;               /* Sweep length (ms)    */
    short  sweep_type;              /* Sweep type     */
    short  sweep_tap_start;         /* Sweep trace taper length at start ms  */
    short  sweep_tap_end;           /* Sweep trace taper length at end ms  */
    short  taper_type;              /* taper type  */ 
    short  filter_fr;               /* Alias filter frequency     */
    short  filter_slope;            /* Alias filter slope     */
    short  notch_fr;                /* Notch filter frequency   */
    short  notch_slope;             /* Notch filter slope    */
    short  low_cut_fr;              /* Low cut frequency    */
    short  high_cut_fr;             /* High cut frequency    */
    short  low_cut_slope;           /* Low cut slope    */
    short  high_cut_slope;          /* High cut slope    */
    short  year ;                   /* Year data recorded      */
    short  day;                     /* Day of year    */
    short  hour;                    /* Hour of day    */
    short  minute;                  /* Minute of hour */
    short  second;                  /* Second of minute */  
    short  time_code;               /* Time basis code      */
    short  weight;                  /* Trace weighting factor   */
    short  geoph_roll_sw_1;         /* Geophone grp. no. of roll switch 1     */
    short  geoph_trace_1;          /* Geophone grp. no. of trace 1 (field)  */
    short  geoph_trace_last;       /* Geophone grp. no. of last trace */
    short  gap;                    /* gap size */
    short  overtravel;             /* Overtravel */
    short  undefined[30];
    char   values[1];
    };
/** GLOBAL PROCEDURE *******************************************************/

void ebcdic_to_ascii_( src, dst, n)
char *src;
char *dst;
int  *n;
  /* This converts the ascii format source character to an EBCDIC format
  destination character. */
{
  int i ;
  for (i=0;i<*n;i++)
    dst[i] = ebc2asc[(unsigned char)src[i]] ;
} ;


/*  *********************************************************************** */
/*     Output seg-y header                                                 */

header_output_( handle_ptr,char_hdr,char_hdr_len,bin_lng,bin_lng_len,bin_sh,
bin_sh_len )
int   *handle_ptr;
char   *char_hdr;
int    *char_hdr_len;
long   *bin_lng;
int    *bin_lng_len;
short  *bin_sh;
int    *bin_sh_len;
  {
  int i;
  int size;
  struct reel_hdr *headr;
  printf( " Writing reel header" );
  printf( " %s ", char_hdr );
  printf( " %u  %u", bin_lng[0], bin_sh[0]);
  printf( " %u  %u", *bin_lng, *bin_sh);
  headr = ( struct reel_hdr * ) malloc( sizeof( struct reel_hdr ));
  
  for ( i = 0; i < 3200; i++ )
    {
    headr->header[i]        = char_hdr[i]; 
    }  
  headr->job_id        = bin_lng[0];              /* Job id number  */
  headr->line_no       = bin_lng[1]; 
  headr->reel_no       = bin_lng[2];              /* Reel number      */
  headr->n_trace       = bin_sh[0];             /* no of traces per record */
  headr->n_auxtr       = bin_sh[1];             /* no of auxiliary traces */
  headr->time_int      = bin_sh[2];          /* sample interval (micro s ) */
  headr->field_int     = bin_sh[3];         /* interval in field record   */
  headr->n_samp        = bin_sh[4];         /* no of samples per trace    */
  headr->field_n_samp  = bin_sh[5];        /* no of samples/tr in field  */
  headr->format        = bin_sh[6];          /* format code                */
  headr->cdp_fold      = bin_sh[7];    /* no of data traces/ cdp ensemble */
  headr->tr_sort_code  = bin_sh[8];            /* trace sorting code */
  headr->vert_sum_code = bin_sh[9];           /* vertical sum code    */
  headr->sweep_fr_st   = bin_sh[10];           /* sweep frequency at start */
  headr->sweep_fr_end  = bin_sh[11];            /* sweep frequency at end*/
  headr->sweep_len     = bin_sh[12];               /* sweep length (ms) */
  headr->sweep_type    = bin_sh[13];              /* sweep type code */
  headr->sweep_tr_no   = bin_sh[14];             /* trace no of sweep ch */
  headr->sweep_tap_st  = bin_sh[15];  
  headr->sweep_tap_end  = bin_sh[16];  /* sweep tr taper length at end (ms) */
  headr->sweep_tap_typ  = bin_sh[17];           /* taper type code */
  headr->correlated     = bin_sh[18];              /* correlated data code */
  headr->bin_gain       = bin_sh[19];        /* binary gain recovered code */
  headr->amp_code       = bin_sh[20];        /* amplitude recovery method */
  headr->meas_code      = bin_sh[21];               /* measurement system */
  headr->sign_pol       = bin_sh[22];          /* impulse signal polarity */
  headr->pol_code       = bin_sh[23];         /*  vibratory polarity code */
  for ( i =0 ; i < 170; i++ )
    {
    headr->unused[i]      = bin_sh[24+i]; 
    }          
  printf( "%s", headr->header );
  printf( "%u %u %u ", headr->job_id, headr->line_no, headr->reel_no);
  printf( " %u %u %u ", headr->n_trace, headr->time_int, headr->n_samp );
  size = *char_hdr_len + *bin_lng_len + *bin_sh_len;
  block_write_( handle_ptr, (char *)headr, &size ); 
  return;
}

/*  *********************************************************************** */
/*     Output seg-y trace                                                */

trace_output_( handle_ptr,lng1,lng1_len,sh1,sh1_len,lng2,lng2_len,
               sh2,sh2_len,lng3,lng3_len,sh3,sh3_len,trace_val,trace_len)
int   *handle_ptr;
long   *lng1;
int    *lng1_len;
short  *sh1;
int    *sh1_len;
long   *lng2;
int    *lng2_len;
short  *sh2;
int    *sh2_len;
long   *lng3;
int    *lng3_len;
short  *sh3;
int    *sh3_len;
char   *trace_val;
int    *trace_len;

  {
  int i;
  int size;
  int limit;

  struct trace *trace;
  printf( " Writing trace " );
  size = *lng1_len + *lng2_len + *lng3_len + 
          *sh1_len + *sh2_len  + *sh3_len  + *trace_len;
  if ( size != sizeof( struct trace ) - 4 + *trace_len )
      printf ( "ERROR: size = %u, trace structure = %u, values = %u ",
                       size,  sizeof( struct trace ), *trace_len );  
  trace = ( struct trace * ) malloc(  size );
    trace->trace_no = lng1[0];   /* Trace no within line    */
    trace->trace_reel = lng1[1]; /* Trace no within reel      */
    trace->field_rec = lng1[2];  /* Original field record no     */
    trace->field_trace = lng1[3];  /* Trace no within field record */
    trace->source_no = lng1[4];     /* Energy source point no */
    trace->cdp_no = lng1[5];          /* CDP ensemble no */
    trace->trace_cdp = lng1[6];               /* Trae no within CDP */
    trace->trace_id = sh1[0];                /* Trace identification code */
    trace->vert_sum_no = sh1[1];         /* No of vertically summed traces  */
    trace->hort_stack_no = sh1[2];      /* No of horizontally stacked traces  */
    trace->data_use = sh1[3];                /* 1=production; 2=test */
    trace->distance = lng2[0];       /* distance from source to receiver  */
    trace->rcvr_elev = lng2[1];               /* receiver group elevation */
    trace->src_elev = lng2[2];                /* source elevation */
    trace->src_depth = lng2[3];               /* source depth */
    trace->rcvr_datum_elev = lng2[4];         /* datum elevation at receiver*/
    trace->src_datum_elev = lng2[5];          /* datum elevation at source */
    trace->src_water_depth = lng2[6];               /* water depth at source*/
    trace->rcvr_water_depth = lng2[7];              /* water depth at receiver*/
    trace->elev_scaler = sh2[0];            /* scaler for elevs. and depths  */
    trace->co_ord_scaler = sh2[1];           /* scaler for co-ords */
    trace->x_src = lng3[0];                   /* x co-ord at source */
    trace->y_src = lng3[1];                   /* y co-ord at source */
    trace->x_rcvr = lng3[2];                  /* x co-ord at receiver */
    trace->y_rcvr = lng3[3];                  /* y co-ord at receiver */
    trace->units = sh3[0];                   /* co-ordinate units */
    trace->sed_vel = sh3[1];                 /* weathering velocity */
    trace->base_vel = sh3[2];                /* sub_weathering velocity */
    trace->time_src = sh3[3];                /* Uphole time at source */
    trace->time_rcvr = sh3[4];               /* Uphole time at receiver */
    trace->static_src = sh3[5];              /* Source static correction */
    trace->static_rcvr = sh3[6];            /* Group static correction */
    trace->static_tot = sh3[7];              /* Total static applied */
    trace->lag_A = sh3[8];                   /* Lag time A */
    trace->lag_B = sh3[9];                   /* Lag time B */
    trace->delay = sh3[10];                   /* Delay recording time   */
    trace->mute_start = sh3[11];              /* Mute time - start   */
    trace->mute_end = sh3[12];                /* Mute time - end  */
    trace->nsamps = sh3[13];                  /* No. of samples in this trace */
    trace->intvl = sh3[14];                   /* Sample interval (microsecs)  */
    trace->gain_type = sh3[15];            /* Gain type of field instruments  */
    trace->gain_const = sh3[16];             /* Instrument gain constant      */
    trace->gain_initial = sh3[17];           /* Instrument initial gain (dB)  */
    trace->correlated = sh3[18];           /* Correlated ( 1 = yes, 2 = no )  */
    trace->sweep_fr_start = sh3[19];          /* Sweep frequency at start     */
    trace->sweep_fr_end = sh3[20];            /* Sweep frequency at end     */
    trace->sweep_len = sh3[21];               /* Sweep length (ms)    */
    trace->sweep_type = sh3[22];              /* Sweep type     */
    trace->sweep_tap_start = sh3[23];/* Sweep trace taper length at start ms  */
    trace->sweep_tap_end = sh3[24];    /* Sweep trace taper length at end ms  */
    trace->taper_type = sh3[25];              /* taper type  */ 
    trace->filter_fr = sh3[26];               /* Alias filter frequency     */
    trace->filter_slope = sh3[27];            /* Alias filter slope     */
    trace->notch_fr = sh3[28];                /* Notch filter frequency   */
    trace->notch_slope = sh3[29];             /* Notch filter slope    */
    trace->low_cut_fr = sh3[30];              /* Low cut frequency    */
    trace->high_cut_fr = sh3[31];             /* High cut frequency    */
    trace->low_cut_slope = sh3[32];           /* Low cut slope    */
    trace->high_cut_slope = sh3[33];          /* High cut slope    */
    trace->year  = sh3[34];                   /* Year data recorded      */
    trace->day = sh3[35];                     /* Day of year    */
    trace->hour = sh3[36];                    /* Hour of day    */
    trace->minute = sh3[37];                  /* Minute of hour */
    trace->second = sh3[38];                  /* Second of minute */  
    trace->time_code = sh3[39];               /* Time basis code      */
    trace->weight = sh3[40];                  /* Trace weighting factor   */
    trace->geoph_roll_sw_1 = sh3[41];   /* Geophone grp. no. of roll switch 1 */
    trace->geoph_trace_1 = sh3[42];  /* Geophone grp. no. of trace 1 (field)  */
    trace->geoph_trace_last = sh3[43];   /* Geophone grp. no. of last trace */
    trace->gap = sh3[44];                    /* gap size */
    trace->overtravel = sh3[45];  
    for ( i =0 ; i < 30; i++ )
      {
      trace->undefined[i]      = sh3[46+i]; 
      }    
/*    printf ("Copied header" );    */
    limit = *trace_len;
    for ( i =0 ; i < limit; i++ )
      {
      trace->values[i]= trace_val[i]; 
      }          
/*    printf ( "Copied values" );  */
  block_write_( handle_ptr, (char *)trace, &size ); 
  return;
}


/*  *********************************************************************** */
/*     Input seg-y reel header                                                 */

header_input_(handle_ptr,char_hdr,char_hdr_len,bin_lng,bin_lng_len,
              bin_sh, bin_sh_len ) 
int  *handle_ptr;            /* file number */
char   *char_hdr;            /* space to return character header */
int    *char_hdr_len;        /* number of bytes in header (3200) */
long   *bin_lng;             /* space for long ints in binary header */
int    *bin_lng_len;         /* number of bytes for longs (12) */ 
short  *bin_sh;              /* space for short ints from binary header */
int    *bin_sh_len;         /* number of bytes for shorts (388 ) */
  {
  int i;
  int size;

  struct reel_hdr *headr;
/*  printf( " Reading reel header" );   */

  headr = ( struct reel_hdr * ) malloc( sizeof( struct reel_hdr ));
  size = *char_hdr_len + *bin_lng_len + *bin_sh_len;
  if ( size != sizeof( struct reel_hdr ) )
      printf("ERROR: sizes wrong: size = %u, struct reel_hdr = %u ",
                                  size,     sizeof( struct reel_hdr ));
  block_read_( handle_ptr, (char *)headr, &size ); 


  bin_lng[0]        = headr->job_id;              /* Job id number  */
  bin_lng[1]        = headr->line_no; 
  bin_lng[2]        = headr->reel_no;              /* Reel number      */
  bin_sh[0]         = headr->n_trace ;             /* no of traces per record */
  bin_sh[1]        = headr->n_auxtr;             /* no of auxiliary traces */
  bin_sh[2]       = headr->time_int;          /* sample interval (micro s ) */
  bin_sh[ 3]        =headr->field_int ;         /* interval in field record   */
  bin_sh[ 4]       = headr->n_samp;         /* no of samples per trace    */
  bin_sh[5]  = headr->field_n_samp;        /* no of samples/tr in field  */
  bin_sh[ 6]       = headr->format;          /* format code                */
  bin_sh[7]      = headr->cdp_fold;    /* no of data traces/ cdp ensemble */
  bin_sh[8]  = headr->tr_sort_code;            /* trace sorting code */
  bin_sh[ 9]= headr->vert_sum_code;           /* vertical sum code    */
  bin_sh[ 10]  = headr->sweep_fr_st;           /* sweep frequency at start */
  bin_sh[11]  = headr->sweep_fr_end;            /* sweep frequency at end*/
  bin_sh[12]     = headr->sweep_len;               /* sweep length (ms) */
  bin_sh[ 13]  = headr->sweep_type ;              /* sweep type code */
  bin_sh [14]  = headr->sweep_tr_no;             /* trace no of sweep ch */
  bin_sh[ 15] = headr->sweep_tap_st;  
  bin_sh[  16]= headr->sweep_tap_end;  /* sweep tr taper length at end (ms) */
  bin_sh[ 17]  =headr->sweep_tap_typ;           /* taper type code */
  bin_sh[ 18]    = headr->correlated;              /* correlated data code */
  bin_sh[19]       = headr->bin_gain;        /* binary gain recovered code */
  bin_sh[ 20]      = headr->amp_code;        /* amplitude recovery method */
  bin_sh[  21]    = headr->meas_code;               /* measurement system */
  bin_sh[22]       =headr->sign_pol;          /* impulse signal polarity */
  bin_sh[23]       = headr->pol_code;         /*  vibratory polarity code */
  for ( i =0 ; i < 170; i++ )
    {
    bin_sh[24+i]      = headr->unused[i]; 
    }          
/*  printf( " /n %s ", char_hdr );   */
/*  printf( " %u  %u", bin_lng[0], bin_sh[0]); */

  return;
}

/*  *********************************************************************** */
/*     Input seg-y trace                                                */

trace_input_( handle_ptr,lng1,lng1_len,sh1,sh1_len,lng2,lng2_len,
              sh2,sh2_len, lng3,lng3_len,sh3,sh3_len,trace_val,trace_len)
int   *handle_ptr;
long   *lng1;
int    *lng1_len;
short  *sh1;
int    *sh1_len;
long   *lng2;
int    *lng2_len;
short  *sh2;
int    *sh2_len;
long   *lng3;
int    *lng3_len;
short  *sh3;
int    *sh3_len;
char   *trace_val;
int    *trace_len;
  {
  int i;
  int size;
  int limit;

  struct trace *trace;
/*  printf( " Reading trace " );*/
  size = *lng1_len + *lng2_len + *lng3_len + 
          *sh1_len + *sh2_len  + *sh3_len  + *trace_len;
  if ( size != sizeof( struct trace ) - 4 + *trace_len ) 
      printf ( "ERROR: size = %u, trace structure = %u, values = %u ",
                       size,  sizeof( struct trace ), *trace_len );  
  trace = ( struct trace * ) malloc(  size );
  block_read_( handle_ptr, (char *)trace, &size );
    lng1[0] = trace->trace_no;   /* Trace no within line    */
    lng1[1] = trace->trace_reel; /* Trace no within reel      */
    lng1[2] = trace->field_rec;  /* Original field record no     */
    lng1[3] = trace->field_trace;  /* Trace no within field record */
    lng1[4] = trace->source_no;     /* Energy source point no */
    lng1[5] = trace->cdp_no;          /* CDP ensemble no */
    lng1[6] = trace->trace_cdp;               /* Trae no within CDP */
    sh1[0] = trace->trace_id;                /* Trace identification code */
    sh1[1] = trace->vert_sum_no;         /* No of vertically summed traces  */
    sh1[2] = trace->hort_stack_no;      /* No of horizontally stacked traces  */
    sh1[3] = trace->data_use;                /* 1=production; 2=test */
    lng2[0] = trace->distance;       /* distance from source to receiver  */
    lng2[1] = trace->rcvr_elev;               /* receiver group elevation */
    lng2[2] = trace->src_elev;                /* source elevation */
    lng2[3] = trace->src_depth;               /* source depth */
    lng2[4] = trace->rcvr_datum_elev;         /* datum elevation at receiver*/
    lng2[5] = trace->src_datum_elev;          /* datum elevation at source */
    lng2[6] = trace->src_water_depth;               /* water depth at source*/
    lng2[7] = trace->rcvr_water_depth;              /* water depth at receiver*/
    sh2[0] = trace->elev_scaler;            /* scaler for elevs. and depths  */
    sh2[1] = trace->co_ord_scaler;           /* scaler for co-ords */
    lng3[0] = trace->x_src;                   /* x co-ord at source */
    lng3[1] = trace->y_src;                   /* y co-ord at source */
    lng3[2] = trace->x_rcvr;                  /* x co-ord at receiver */
    lng3[3] = trace->y_rcvr;                  /* y co-ord at receiver */
    sh3[0] = trace->units;                   /* co-ordinate units */
    sh3[1] = trace->sed_vel;                 /* weathering velocity */
    sh3[2] = trace->base_vel;                /* sub_weathering velocity */
    sh3[3] = trace->time_src;                /* Uphole time at source */
    sh3[4] = trace->time_rcvr;               /* Uphole time at receiver */
    sh3[5] = trace->static_src;              /* Source static correction */
    sh3[6] = trace->static_rcvr;            /* Group static correction */
    sh3[7] = trace->static_tot;              /* Total static applied */
    sh3[8] = trace->lag_A;                   /* Lag time A */
    sh3[9] = trace->lag_B;                   /* Lag time B */
    sh3[10] = trace->delay;                   /* Delay recording time   */
    sh3[11] = trace->mute_start;              /* Mute time - start   */
    sh3[12] = trace->mute_end;                /* Mute time - end  */
    sh3[13] = trace->nsamps;                  /* No. of samples in this trace */
    sh3[14] = trace->intvl;                   /* Sample interval (microsecs)  */
    sh3[15] = trace->gain_type;            /* Gain type of field instruments  */
    sh3[16] = trace->gain_const;             /* Instrument gain constant      */
    sh3[17] = trace->gain_initial;           /* Instrument initial gain (dB)  */
    sh3[18] = trace->correlated;           /* Correlated ( 1 = yes, 2 = no )  */
    sh3[19] = trace->sweep_fr_start;          /* Sweep frequency at start     */
    sh3[20] = trace->sweep_fr_end;            /* Sweep frequency at end     */
    sh3[21] = trace->sweep_len;               /* Sweep length (ms)    */
    sh3[22] = trace->sweep_type;              /* Sweep type     */
    sh3[23] = trace->sweep_tap_start;/* Sweep trace taper length at start ms  */
    sh3[24] = trace->sweep_tap_end;    /* Sweep trace taper length at end ms  */
    sh3[25] = trace->taper_type;              /* taper type  */ 
    sh3[26] = trace->filter_fr;               /* Alias filter frequency     */
    sh3[27] = trace->filter_slope;            /* Alias filter slope     */
    sh3[28] = trace->notch_fr;                /* Notch filter frequency   */
    sh3[29] = trace->notch_slope;             /* Notch filter slope    */
    sh3[30] = trace->low_cut_fr;              /* Low cut frequency    */
    sh3[31] = trace->high_cut_fr;             /* High cut frequency    */
    sh3[32] = trace->low_cut_slope;           /* Low cut slope    */
    sh3[33] = trace->high_cut_slope;          /* High cut slope    */
    sh3[34] = trace->year;                   /* Year data recorded      */
    sh3[35] = trace->day;                     /* Day of year    */
    sh3[36] = trace->hour;                    /* Hour of day    */
    sh3[37] = trace->minute;                  /* Minute of hour */
    sh3[38] = trace->second;                  /* Second of minute */  
    sh3[39] = trace->time_code;               /* Time basis code      */
    sh3[40] = trace->weight;                  /* Trace weighting factor   */
    sh3[41] = trace->geoph_roll_sw_1;   /* Geophone grp. no. of roll switch 1 */
    sh3[42] = trace->geoph_trace_1;  /* Geophone grp. no. of trace 1 (field)  */
    sh3[43] = trace->geoph_trace_last;   /* Geophone grp. no. of last trace */
    sh3[44] = trace->gap;                    /* gap size */
    sh3[45] = trace->overtravel;  
    for ( i =0 ; i < 30; i++ )
      {
       sh3[46+i] = trace->undefined[i]; 
      }    
/*    printf ("Copied header" );   */
/*    limit = (*trace_len)/4;      */
    limit = *trace_len;
    for ( i =0 ; i < limit; i++ )
      {
      trace_val[i] = trace->values[i]; 
      }          
/*    printf ( "Copied values" );   */
 
  return;
}
