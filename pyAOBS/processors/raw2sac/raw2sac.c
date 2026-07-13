/*----------------------------------------------------------------------
 *
	program name:	raw2sac.c 
	written by:	XL Qiu, MH Zhao and SH Xia     
	first attempt:	2008/05/30
        modified:       2009/03/16  add SPS input from the command line
        modified:       2009/04/29  merge rawb2sac and raws2sac
        modified:       2009/06/01  output npts-1, for merge in SAC to work
    
	This program converts the new OBS raw data to SAC format
	
	usage1: raw2sac <infile> <sps>
	usage2: raw2sac <infile>       (default sps=125)
 
*******************************************************************************/
#include "stdio.h"      /* Standard I/O header file  */
#include "stdlib.h"
#include "sac_new.h"
#include "math.h"
#include "string.h"
#include "ctype.h"

struct sac sp;
long julian_day(long year, long mon, long day);
unsigned int HextoInt(unsigned char datebyte);

 struct
   { char hbyt1,mbyt1,lbyt1;
     char hbyt2,mbyt2,lbyt2;
     char hbyt3,mbyt3,lbyt3;
     char hbyt4,mbyt4,lbyt4;
        } data1;

 struct
   { char hbyt1,mbyt1,lbyt1;
     char hbyt2,mbyt2,lbyt2;
     char hbyt3,mbyt3,lbyt3;
        } data2;

 union
   { unsigned char byteval[4];
     short tbytev[2];
     long fbytev;
     } threeB2long;
 
main (argc,argv)
int argc;
char  *argv[];

{ FILE *fpin;               	 /* pointer to the input file */  
  FILE *fp1,*fp2,*fp3,*fp4;        /* pointer to the output file */ 

  unsigned long int i, ifl;
  unsigned char fnamein[64],fnameout[13];
  unsigned int date[8],msec[3],fnlength;
  unsigned int mon,day,sps;
  long int jday,sacday;
  float ftd;

  if (argc<2)
   {printf("you forgot to enter a filename\n");
    printf("usage1: raw2sac <infile> <sps>\n");
    printf("usage2: raw2sac <infile>   (default sps=125)\n");
    exit(0);
    }

  strcpy (fnamein, argv[1]);

  if (argc<3)
   {printf("you forgot the SPS, set default 125\n");
    sps=125;
    }
  else
    sscanf(argv[2],"%d",&sps); 

  if ((fpin=fopen(fnamein,"r"))==NULL)
   {printf("cannot open infile\n");
    exit(0);
   }

  fnlength = strlen(fnamein);

  for (i=0;i<9;i++) fnameout[i] = fnamein[fnlength-12+i];

  /* initiate binary sac header  */ 

  sp = sac_null;

  /* get the date and time information for the file name */

  for (i=0;i<8;i++) date[i] = HextoInt(fnamein[fnlength-12+i]);

  sp.nzyear = (date[0] << 2 | date[1] >> 2)  & 0x0f;  /* 0x0f=00001111 */
        mon = (date[1] << 2 | date[2] >> 2)  & 0x0f;  /* 0x0f=00001111 */
        day = (date[2] << 3 | date[3] >> 1)  & 0x1f;  /* 0x1f=00011111 */
  sp.nzhour = (date[3] << 4 | date[4])       & 0x1f;  /* 0x1f=00011111 */
   sp.nzmin = (date[5] << 2 | date[6] >> 2)  & 0x3f;  /* 0x3f=00111111 */
   sp.nzsec = (date[6] << 4 | date[7])       & 0x3f;  /* 0x3f=00111111 */

  sp.nzyear =  sp.nzyear + 2000;

  printf("%4ld-%02d-%02d\n", sp.nzyear, mon, day);

  jday=julian_day((long)sp.nzyear,mon,day);
  sacday=julian_day((long)sp.nzyear,1,1);
  sp.nzjday=jday-sacday+1;

  for (i=9;i<12;i++) msec[i-9] = HextoInt(fnamein[fnlength-12+i]);

  sp.nzmsec = (msec[0] << 8 | msec[1] << 4 | msec[2]);
  sp.nzmsec = (long)(sp.nzmsec/2.048 + 0.5);

  printf("%02ld:%02ld:%06.3f UTC\n", sp.nzhour, sp.nzmin, sp.nzsec+sp.nzmsec/1000.);

  fnameout[12]='\0';

 if(date[0]>>3 == 0) 
  {

  fnameout[9]='b';
  fnameout[10]='h';
  fnameout[11]='x';

  if ((fp1=fopen(fnameout,"wb"))==NULL)
   {printf("the output file1 can not open\n");
    exit(0);
   }

  fnameout[11]='y';

  if ((fp2=fopen(fnameout,"wb"))==NULL)
   {printf("the output file2 can not open\n");
    exit(0);
   }

  fnameout[11]='z';

  if ((fp3=fopen(fnameout,"wb"))==NULL)
   {printf("the output file3 can not open\n");
    exit(0);
   }

  fnameout[9]='h';
  fnameout[10]='y';
  fnameout[11]='d';

  if ((fp4=fopen(fnameout,"wb"))==NULL)
   {printf("the output file4 can not open\n");
    exit(0);
   }


  /* check the file length and get the number of data points */

  fseek(fpin,0L,2);
  ifl=ftell(fpin);
  rewind(fpin);

  sp.npts=ifl/sizeof(data1); /* output npts-1, for merge in SAC to work */

  printf("npts=%ld\n", sp.npts);

  /* setup other sac header parameters    */
 
  
  sp.delta = 1./sps;
  sp.b = 0.0;

  sp.iftype = ITIME;
  sp.iztype = IB;

  sp.leven = TRUE;
  sp.lcalda = TRUE;

  strcpy(sp.kstnm,"OBS");

  /*write binary sac header  */

  sp.cmpinc=90.0;
  strcpy(sp.kcmpnm,"BHX");

  if(fwrite(&sp,sizeof(struct sac),1,fp1)!=1)
   {
   printf("FATAL ERROR:bad write%s,header\n",fp1);
      exit(4);
   }

  sp.cmpinc=90.0;
  strcpy(sp.kcmpnm,"BHY");

  if(fwrite(&sp,sizeof(struct sac),1,fp2)!=1)
   {
   printf("FATAL ERROR:bad write%s,header\n",fp2);
      exit(4);
   }

  sp.cmpinc=0.0;
  sp.cmpaz=0.0;
  strcpy(sp.kcmpnm,"BHZ");

  if(fwrite(&sp,sizeof(struct sac),1,fp3)!=1)
   {
   printf("FATAL ERROR:bad write%s,header\n",fp3);
      exit(4);
   }

  sp.cmpinc=12345.;
  sp.cmpaz=12345.;
  strcpy(sp.kcmpnm,"HYD");

  if(fwrite(&sp,sizeof(struct sac),1,fp4)!=1)
   {
   printf("FATAL ERROR:bad write%s,header\n",fp4);
      exit(4);
   }

  /*write binary sac data  */

  for (i=0;i<sp.npts;i++)
  {

  /* read the raw data */

  fread(&data1,sizeof(data1),1,fpin);

  /* convert and  write the ch1 raw data */

  threeB2long.byteval[0]= data1.lbyt1;  /* swap the bytes */
  threeB2long.byteval[1]= data1.mbyt1;
  threeB2long.tbytev[1]= (short)data1.hbyt1;  /*sign bit extension*/

  ftd = (float)threeB2long.fbytev;
  fwrite(&ftd,sizeof(float),1,fp1);

  /* convert and  write the ch2 raw data */

  threeB2long.byteval[0]= data1.lbyt2;  /* swap the bytes */
  threeB2long.byteval[1]= data1.mbyt2;
  threeB2long.tbytev[1]= (short)data1.hbyt2;  /*sign bit extension*/

  ftd = (float)threeB2long.fbytev;
  fwrite(&ftd,sizeof(float),1,fp2);

  /* convert and write the ch3 raw data */

  threeB2long.byteval[0]= data1.lbyt3;  /* swap the bytes */
  threeB2long.byteval[1]= data1.mbyt3;
  threeB2long.tbytev[1]= (short)data1.hbyt3;  /*sign bit extension*/

  ftd = (float)threeB2long.fbytev;
  fwrite(&ftd,sizeof(float),1,fp3);

  /* convert and write the ch4 raw data */

  threeB2long.byteval[0]= data1.lbyt4;  /* swap the bytes */
  threeB2long.byteval[1]= data1.mbyt4;
  threeB2long.tbytev[1]= (short)data1.hbyt4;  /*sign bit extension*/

  ftd = (float)threeB2long.fbytev;
  fwrite(&ftd,sizeof(float),1,fp4);

  }

  fclose(fpin);
  fclose(fp1);
  fclose(fp2);
  fclose(fp3);
  fclose(fp4);
  }
 else
  {

  fnameout[9]='s';
  fnameout[10]='h';
  fnameout[11]='x';

  if ((fp1=fopen(fnameout,"wb"))==NULL)
   {printf("the output file1 can not open\n");
    exit(0);
   }

  fnameout[11]='y';

  if ((fp2=fopen(fnameout,"wb"))==NULL)
   {printf("the output file2 can not open\n");
    exit(0);
   }

  fnameout[11]='z';

  if ((fp3=fopen(fnameout,"wb"))==NULL)
   {printf("the output file3 can not open\n");
    exit(0);
   }

  /* check the file length and get the number of data points */

  fseek(fpin,0L,2);
  ifl=ftell(fpin);
  rewind(fpin);

  sp.npts=ifl/sizeof(data2); /* output npts-1, for merge in SAC to work */

  printf("npts=%ld\n", sp.npts);

  /* setup other sac header parameters    */
 
  sp.delta = 1./sps;
  sp.b = 0.0;

  sp.iftype = ITIME;
  sp.iztype = IB;

  sp.leven = TRUE;
  sp.lcalda = TRUE;

  strcpy(sp.kstnm,"OBS");

  /*write binary sac header  */

  sp.cmpinc=90.0;
  strcpy(sp.kcmpnm,"SHX");

  if(fwrite(&sp,sizeof(struct sac),1,fp1)!=1)
   {
   printf("FATAL ERROR:bad write%s,header\n",fp1);
      exit(4);
   }

  sp.cmpinc=90.0;
  strcpy(sp.kcmpnm,"SHY");

  if(fwrite(&sp,sizeof(struct sac),1,fp2)!=1)
   {
   printf("FATAL ERROR:bad write%s,header\n",fp2);
      exit(4);
   }

  sp.cmpinc=0.0;
  sp.cmpaz=0.0;
  strcpy(sp.kcmpnm,"SHZ");

  if(fwrite(&sp,sizeof(struct sac),1,fp3)!=1)
   {
   printf("FATAL ERROR:bad write%s,header\n",fp3);
      exit(4);
   }
/*
  sp.cmpinc=12345.;
  sp.cmpaz=12345.;
  strcpy(sp.kcmpnm,"H");

 if(fwrite(&sp,sizeof(struct sac),1,fp4)!=1)
  {
   printf("FATAL ERROR:bad write%s,header\n",fp4);
      exit(4);
   }
*/

  /*write binary sac data  */

  for (i=0;i<sp.npts;i++)
  {

  /* read the raw data */

  fread(&data2,sizeof(data2),1,fpin);

  /* convert and  write the ch1 raw data */

  threeB2long.byteval[0]= data2.lbyt1;  /* swap the bytes */
  threeB2long.byteval[1]= data2.mbyt1;
  threeB2long.tbytev[1]= (short)data2.hbyt1;  /*sign bit extension*/

  ftd = (float)threeB2long.fbytev;
  fwrite(&ftd,sizeof(float),1,fp1);

  /* convert and  write the ch2 raw data */

  threeB2long.byteval[0]= data2.lbyt2;  /* swap the bytes */
  threeB2long.byteval[1]= data2.mbyt2;
  threeB2long.tbytev[1]= (short)data2.hbyt2;  /*sign bit extension*/

  ftd = (float)threeB2long.fbytev;
  fwrite(&ftd,sizeof(float),1,fp2);

  /* convert and write the ch3 raw data */

  threeB2long.byteval[0]= data2.lbyt3;  /* swap the bytes */
  threeB2long.byteval[1]= data2.mbyt3;
  threeB2long.tbytev[1]= (short)data2.hbyt3;  /*sign bit extension*/

  ftd = (float)threeB2long.fbytev;
  fwrite(&ftd,sizeof(float),1,fp3);

  }

  fclose(fpin);
  fclose(fp1);
  fclose(fp2);
  fclose(fp3);


  }

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
