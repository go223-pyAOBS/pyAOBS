/*----------------------------------------------------------------------
 *
	program name:	fntime.c 
	written by:	XL Qiu   
	first attempt:	2009/4/26
        modified by: JZ Zhang, HY Zhang
        first attempt:  2015/11/9
        modified:       2015/11/09  by JZ Zhang, new output for nzmsec
		modified:       2016/04/21  by XL Qiu, set the year valid for 2009-2024
        modified:       2016/08/31  by HY Zhang, add TC,sps_pre as new input parameters and add gap(millisecondd), real delta(millisecond)
                                                 in output.
												 
	note: applied to portable types, e.g. type A,B,L,D,S,etc;
		      TC is read from A*.LOG file;
			  In general, TC of same type of OBSs at the same theoretical sampling rate 
			  can be considered same in this code;
			  TC is just used when parsing millisecond of starting time;
			  sps_pre is the preset sampling rate of OBS;
        
	
	usage: fntime_v1.1 fileNameList TC sps_pre
    TC is read from A*.LOG file.
*******************************************************************************/
#include "stdio.h"      /* Standard I/O header file  */
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "ctype.h"

long julian_day(long year, long mon, long day);
unsigned int HextoDec(unsigned int a0, unsigned char *p);
unsigned int HextoInt(unsigned char datebyte);


 
main (argc,argv)
int argc;
char  *argv[];

{ FILE *fpin;               	 /* pointer to the input file */  
  FILE *fpout;       		 /* pointer to the output file */ 
  FILE *fpdata;       		 /* pointer to the data file */ 

  unsigned long int i, ifl;
  unsigned char fnlist[64],linefd[5],fnamein[64],fnamein1[64],fnameout[30],c[30],msec0[3],msec01[3];
  unsigned int date[8],msec[4],fnlength, scandata;
  unsigned int year,mon,day,hour,min,sec,a1,nfa,nf2;
  unsigned int year1, mon1, day1, hour1, min1, sec1;
  long int jday,nzmsec,npts;
  long int jday1, nzmsec1, npts1;
  double time, time1, TC, PCIk;
  double sps, sps1, sps_pre, delta;

  
  
  
  if (argc!=4)
   {	
		printf("you forgot to enter a filename list or TC or preset sps\n The correct usage:fntime_v1.1 fileNameList TC sps_pre\n");
		exit(0);
    }

  strcpy (fnlist, argv[1]);
  
	/*get TC of OBS. TC is included in A*.LOG file*/
  sscanf(argv[2],"%lf",&TC); 
  printf("TC:%lf\n",TC);
  /*get preset sps of OBS*/
  sscanf(argv[3], "%lf", &sps_pre);
  /* set the dominant frequency of internal clock (PCIk) */
  PCIk=TC/256.0;
  
  if ((fpin=fopen(fnlist,"r"))==NULL)
   {	
		printf("cannot open infile\n");
		exit(0);
   }

  strcpy (fnameout, "fntime.txt");

  if ((fpout=fopen(fnameout, "wb"))==NULL)
   {	
		printf("the output file can not open\n");
		exit(0);
   }

  /* get the date and time information from the file name */

  nfa=0;
  nf2=0;
  time1=0.0;
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
	  if (year<2009)year=year+16;

	  a1=0;
	  for (i=9;i<12;i++){
		msec0[i-9]=fnamein[i];
		a1++;
	  }
	  printf("****************\n");
	  printf("fileName:%s\n",fnamein);
	  printf("extend fileName:%s\n", msec0);
	  /* printf("%d\n", a1); */
	  printf("a1: %d\n", a1);
	  nzmsec = HextoDec(--a1,msec0);
	  printf("PCIk: %lf\n", PCIk);
	  printf("nzmsec: %d\n", nzmsec);
	  printf("nzmsec*4096/PCIk: %f\n", nzmsec*4096/PCIk);

	  jday= julian_day(year,mon,day);
	  /* Fix: Convert jday to double before calculation to avoid integer overflow
	     on 32-bit systems where long is 32-bit. jday*24*60*60 can exceed 32-bit int max.
	     On Cygwin (32-bit long): ~2.1e9 max, but jday*86400 can be ~2.1e11 for 2020.
	     On WSL (64-bit long): no overflow issue, but explicit cast ensures portability. */
	  time = ((((double)jday*24.0+(double)hour)*60.0+(double)min)*60.0+(double)sec)+nzmsec*4096.0/PCIk;
	  printf("time=%f jday=%d\n", time, jday);

	  /* check the file length and get the number of data points */

	  if ((fpdata=fopen(fnamein,"r"))==NULL)
	   {	
			printf("cannot open the data file %s\n", fnamein);
			exit(0);
	   }

	  fseek(fpdata,0L,2);
	  ifl=ftell(fpdata);
	  fclose(fpdata);

	  if(date[0]>>3 == 0) scandata=12, nfa=nfa+1;  /* 4 channel data, 3 broadband + 1 hydrophone */
	  if(date[0]>>3 == 1) scandata=9, nf2=nf2+1;   /* 3 channels of short period */

	  npts=ifl/scandata;

	  /*
	  if(nfa == 1 || nf2 == 1)
		fprintf(fpout, "%s  %4ld-%02d-%02d %02ld:%02ld:%06.3f UTC npts=%ld\n", 
				fnamein,year,mon,day,hour,min,sec+nzmsec*4096/PCIk,npts);
	  else
	  {
			sps=npts/fabs(time-time1);
			fprintf(fpout, "%s  %4ld-%02d-%02d %02ld:%02ld:%06.3f UTC dt=%10.6lf microsecond, sps=%10.6lf Hz, Gap=%10.6lf millisecond\n", 
				fnamein,year,mon,day,hour,min,sec+nzmsec*4096/PCIk,fabs(time-time1),sps,(1.0/sps - 1.0/sps_pre)*npts*1000.0 );
	  }
	  */
	  
	  if(nfa == 1 || nf2 == 1)
		  //void sentence
			;
	  else
	  {
			delta=fabs(time-time1)/npts1;
			fprintf(fpout, "%s  %4ld-%02d-%02d %02ld:%02ld:%06.3f UTC time_length=%10.6lf second, delta=%10.6lf msec, Gap=%10.6lf msec\n", 
				fnamein1,year1,mon1,day1,hour1,min1,sec1+nzmsec1*4096/PCIk,fabs(time-time1),1000*delta,(1.0/sps_pre - delta)*npts1*1000.0 );
	  }
	  
	  printf("time1: %f\n", time1);
	  time1=time;
	  printf("time: %f\n", time);
	  
	  strcpy(fnamein1, fnamein);
	  strcpy(msec01, msec0);
	  year1 = year;
	  mon1 = mon;
	  day1 = day;
	  hour1 = hour;
	  min1 = min;
	  sec1 = sec;
	  nzmsec1 = nzmsec;
	  npts1 = npts;
	  //sps1 = sps;
	  
	  

 }
 

 fprintf(fpout, "%s  %4ld-%02d-%02d %02ld:%02ld:%06.3f UTC\n", 
				fnamein1,year1,mon1,day1,hour1,min1,sec1+nzmsec1*4096/PCIk);
 
 

  fclose(fpin);
  fclose(fpout);


}    /* main program end  */


long julian_day(long year, long mon, long day)
{return(day-32075+1461*(year+4800-(14-mon)/12)/4+367*(mon-2+(14-mon)/12*12)/12-3
*((year+4900-(14-mon)/12)/100)/4);}

unsigned int HextoDec(unsigned int a0, unsigned char *p)
{
   unsigned int i,j,sum;
   sum=0;
   printf("a0: %d\n", a0);
   for(i=0;i<=a0;i++){
      if(*(p+i)<='f'&&*(p+i)>='a')
           j=(unsigned int)(*(p+i))-87;
      else if(*(p+i)<='F'&&*(p+i)>='A')
           j=(unsigned int)(*(p+i))-55;
      else
           j=(unsigned int)(*(p+i))-48;
      sum=sum+pow(16,a0-i)*j;
   }
   return(sum);
}
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
