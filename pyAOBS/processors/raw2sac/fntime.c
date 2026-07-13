/*----------------------------------------------------------------------
 *
	program name:	fntime.c 
	written by:	XL Qiu   
	first attempt:	2009/4/26
      
	This program checks the OBS data time from the file name list
	
	usage: rawtime <infile>
 
*******************************************************************************/
#include "stdio.h"      /* Standard I/O header file  */
#include "math.h"
#include "string.h"
#include "ctype.h"

long julian_day(long year, long mon, long day);
unsigned int HextoInt(unsigned char datebyte);


 
main (argc,argv)
int argc;
char  *argv[];

{ FILE *fpin;               	 /* pointer to the input file */  
  FILE *fpout;       		 /* pointer to the output file */ 
  FILE *fpdata;       		 /* pointer to the data file */ 

  unsigned long int i, ifl;
  unsigned char fnlist[64],linefd[5],fnamein[64],fnameout[12];
  unsigned int date[8],msec[3],fnlength, scandata;
  unsigned int year,mon,day,hour,min,sec,sps,nfa,nf2;
  long int jday,nzmsec,npts;
  double time, time1;

  if (argc!=2)
   {printf("you forgot to enter a filename\n");
    exit(0);
    }

  strcpy (fnlist, argv[1]);

  if ((fpin=fopen(fnlist,"r"))==NULL)
   {printf("cannot open infile\n");
    exit(0);
   }

  strcpy (fnameout, "fntime.txt");

  if ((fpout=fopen(fnameout, "wb"))==NULL)
   {printf("the output file can not open\n");
    exit(0);
   }

  /* get the date and time information from the file name */

  nfa=0;
  nf2=0;

 while (fgets(fnamein,13,fpin) != NULL)
 {

  fgets(linefd,5,fpin);

  for (i=0;i<8;i++) date[i] = HextoInt(fnamein[i]);

  year = (date[0] << 2 | date[1] >> 2)  & 0x0f;  /* 0x0f=00001111 */
   mon = (date[1] << 2 | date[2] >> 2)  & 0x0f;  /* 0x0f=00001111 */
   day = (date[2] << 3 | date[3] >> 1)  & 0x1f;  /* 0x1f=00011111 */
  hour = (date[3] << 4 | date[4])       & 0x1f;  /* 0x1f=00011111 */
   min = (date[5] << 2 | date[6] >> 2)  & 0x3f;  /* 0x3f=00111111 */
   sec = (date[6] << 4 | date[7])       & 0x3f;  /* 0x3f=00111111 */

  year =  year + 2000;

  for (i=9;i<12;i++) msec[i-9] = HextoInt(fnamein[i]);

  nzmsec = (msec[0] << 8 | msec[1] << 4 | msec[2]);
  nzmsec = (long)(nzmsec/2.048 + 0.5);

  jday= julian_day(year,mon,day);
  time = ((jday*24+hour)*60+min)*60+sec+nzmsec/1000.;

  /* check the file length and get the number of data points */

  if ((fpdata=fopen(fnamein,"r"))==NULL)
   {printf("cannot open the data file %s\n", fnamein);
    exit(0);
   }

  fseek(fpdata,0L,2);
  ifl=ftell(fpdata);
  fclose(fpdata);

  if(date[0]>>3 == 0) scandata=12, nfa=nfa+1;  /* 4 channel data, 3 broadband + 1 hydrophone */
  if(date[0]>>3 == 1) scandata=9, nf2=nf2+1;   /* 3 channels of short period */

  npts=ifl/scandata;

  if(nfa == 1 || nf2 == 1)
    fprintf(fpout, "%s  %4ld-%02d-%02d %02ld:%02ld:%06.3f UTC npts=%ld\n", 
            fnamein,year,mon,day,hour,min,sec+nzmsec/1000.,npts);
  else
  {
    sps=npts/(int)(fabs(time-time1)+0.5);
    fprintf(fpout, "%s  %4ld-%02d-%02d %02ld:%02ld:%06.3f UTC dt=%10.3lf sps=%d\n", 
            fnamein,year,mon,day,hour,min,sec+nzmsec/1000.,fabs(time-time1),sps);
  }
  
  time1=time;

 }

  fclose(fpin);
  fclose(fpout);


}    /* main program end  */


long julian_day(long year, long mon, long day)
{return(day-32075+1461*(year+4800-(14-mon)/12)/4+367*(mon-2+(14-mon)/12*12)/12-3
*((year+4900-(14-mon)/12)/100)/4);}

unsigned int HextoInt(unsigned char datebyte)
{
   unsigned int h;

   if(!isxdigit(datebyte)) 
   {printf("The input filename is not a Hex number\n");
    exit(0);
   }

   if (datebyte < 58)  h= datebyte - 48;
   else h= toupper(datebyte) - 55;

   return(h);
}
