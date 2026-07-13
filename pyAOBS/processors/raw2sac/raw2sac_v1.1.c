/*----------------------------------------------------------------------
 *
	program name:	raw2sac.c 
	written by:	XL Qiu, MH Zhao and SH Xia     
	first attempt:	2008/05/30
        modified:       2009/03/16  add SPS input from the command line
        modified:       2009/04/29  merge rawb2sac and raws2sac
        modified:       2009/06/01  output npts-1, for merge in SAC to work
        modified:       2015/11/09  by JZ Zhang, new output for nzmsec
		modified:       2016/04/21  by XL Qiu, set the year valid for 2009-2024
        modified:       2016/08/31  by HY Zhang, TC is inclued in parameters, 
		                            the extend name of output SAC file is changed 
								to short period type.
								
	This program converts the new OBS raw data to SAC format
        
        note: applied to portable types, e.g. type A,B,L,D,S,etc;
		      TC is read from A*.LOG file;
			  In general, TC of same type of OBSs at the same theoretical sampling rate 
			  can be considered same in this code;
			  TC is just used when parsing millisecond of starting time;
	
	usage: raw2sac_v1.1 fileName sps TC
	
	IMPORTANT: For 64-bit systems (e.g., WSL), compile with:
	           gcc -O2 -fpack-struct raw2sac_v1.1.c -o raw2sac_v1.1 -lm
	           
	           Options:
	           - -O2: Enable optimizations for better performance (recommended)
	           - -fpack-struct: Ensure structure alignment matches SAC format (now handled in code)
	           
	           Performance tips for WSL:
	           - Files on WSL filesystem (/home, /tmp) are faster than Windows filesystem (/mnt/c, /mnt/d)
	           - For large files, copy to WSL filesystem before processing for better I/O performance
	           - Code includes I/O buffering optimizations for faster file operations
 
*******************************************************************************/

/* Force structure packing to 1 byte alignment for SAC format compatibility */
/* This is critical for 64-bit systems where default alignment would add padding */
#ifdef __GNUC__
#pragma pack(push, 1)  /* Save current packing, set to 1-byte alignment */
#elif defined(_MSC_VER)
#pragma pack(push, 1)  /* MSVC: save current packing, set to 1-byte alignment */
#else
#pragma pack(1)        /* Other compilers: just set 1-byte alignment */
#endif

#include "stdio.h"      /* Standard I/O header file  */
#include "stdlib.h"  
#include "sac_new.h"
#include "math.h"
#include "string.h"
#include "ctype.h"

/* Note: We keep packing=1 for the entire program since struct sac is used throughout.
   If you need to restore default packing for other structures, use:
   #pragma pack(pop)
   But make sure struct sac remains packed! */

struct sac sp;
long julian_day(int year, int mon, int day);  /* Changed return type from int to long */
unsigned int HextoDec(unsigned int a0, unsigned char *p);
unsigned int HextoInt(unsigned char datebyte);

/* Optimize file I/O with larger buffers for better performance on WSL */
static void set_io_buffer(FILE *fp, char *buffer, size_t size) {
  if (buffer != NULL && fp != NULL) {
    setvbuf(fp, buffer, _IOFBF, size);  /* Full buffering for maximum performance */
  }
}

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
     int fbytev;
     } threeB2long;
 
main (argc,argv)
int argc;
char  *argv[];

{ FILE *fpin;               	 /* pointer to the input file */  
  FILE *fp1,*fp2,*fp3,*fp4;        /* pointer to the output file */ 

  unsigned int i, ifl;
  unsigned char fnamein[64],fnameout[13],msec0[3];
  unsigned int date[8],msec[3],fnlength;
  unsigned int mon,day,a1;
  long jday,sacday;  /* Changed from int to long to match SAC header field type (nzjday is long) */
  float ftd,sps;
  double PCIk,TC;

  if (argc==4)
  {
	  strcpy (fnamein, argv[1]);
	  sscanf(argv[2],"%f",&sps); 
	  sscanf(argv[3],"%lf",&TC); 
  }
  else
  {
	  printf("error in usage!\n");
	  printf("correct usage: raw2sac_v1.1 fileName sps TC\n");
	  exit(0);	  
  }
  printf("****************\n");
  printf("fileName:%s\n",fnamein);
  printf("SPS:%f, TC:%lf\n",sps,TC);
  
  /* Check structure size for SAC format compatibility */
  /* SAC file format requires struct sac to be exactly 632 bytes */
  {
    size_t sac_size = sizeof(struct sac);
    if (sac_size != 632) {
      printf("ERROR: struct sac size is %zu bytes, expected 632 bytes!\n", sac_size);
      printf("SAC files generated will be UNREADABLE by SAC software!\n");
      printf("\nTo fix this, ensure the code was compiled with:\n");
      printf("  - Structure packing enabled (code includes #pragma pack(1))\n");
      printf("  - OR compile with: gcc -fpack-struct raw2sac_v1.1.c -o raw2sac_v1.1 -lm\n");
      exit(1);
    } else {
      printf("OK: struct sac size is %zu bytes (correct for SAC format).\n", sac_size);
    }
  }
  
  /* set the dominant frequency of internal clock (PCIk) */
  PCIk=TC/256.;
  printf("PCIk:%lf\n",PCIk);

  if ((fpin=fopen(fnamein,"r"))==NULL)
   {	
		printf("cannot open infile\n");
		exit(0);
   }
  
  /* Optimize file I/O performance: set larger buffer for faster reads */
  /* WSL file I/O performance notes:
   *   - WSL has slower I/O than Cygwin due to Linux->Windows translation layer
   *   - Files on Windows filesystem (/mnt/c, /mnt/d) are slower than WSL filesystem
   *   - Solution: Use larger buffers to reduce system call overhead
   *   - For best performance, copy files to WSL filesystem (/tmp or /home) first
   */
  #define IO_BUFFER_SIZE (256 * 1024)  /* 256 KB buffer */
  char *input_buffer = (char*)malloc(IO_BUFFER_SIZE);
  if (input_buffer != NULL) {
    set_io_buffer(fpin, input_buffer, IO_BUFFER_SIZE);
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
  if (sp.nzyear<2009)sp.nzyear=sp.nzyear+16;
  
  printf("%4ld-%02d-%02d\n", sp.nzyear, mon, day);

  jday=julian_day((int)sp.nzyear,mon,day);
  sacday=julian_day((int)sp.nzyear,1,1);
  sp.nzjday=jday-sacday+1;

  a1=0;
  for (i=9;i<12;i++){
    msec0[i-9]=fnamein[i];
    a1++;
  }
  sp.nzmsec = HextoDec(--a1,msec0);
  printf("msec decimal_code: %" SAC_LONG_FMT "\n", sp.nzmsec);
  sp.nzmsec = (sac_long_t)(sp.nzmsec*4096.*1000./PCIk + 0.5);
  printf("msec: %" SAC_LONG_FMT "\n", sp.nzmsec);
  printf("starting time: %02" SAC_LONG_FMT ":%02" SAC_LONG_FMT ":%06.3f UTC\n", sp.nzhour, sp.nzmin, sp.nzsec+sp.nzmsec/1000.);

  fnameout[12]='\0';

 if(date[0]>>3 == 0) 
  {

	  fnameout[9]='s';
	  fnameout[10]='h';
	  fnameout[11]='x';

	  /* Allocate buffers for output files to improve I/O performance */
	  char *output_buf1 = (char*)malloc(IO_BUFFER_SIZE);
	  char *output_buf2 = (char*)malloc(IO_BUFFER_SIZE);
	  char *output_buf3 = (char*)malloc(IO_BUFFER_SIZE);
	  char *output_buf4 = (char*)malloc(IO_BUFFER_SIZE);

	  if ((fp1=fopen(fnameout,"wb"))==NULL)
	   {	printf("the output file1 can not open\n");
			exit(0);
	   }
	  set_io_buffer(fp1, output_buf1, IO_BUFFER_SIZE);

	  fnameout[11]='y';

	  if ((fp2=fopen(fnameout,"wb"))==NULL)
	   {	
			printf("the output file2 can not open\n");
			exit(0);
	   }
	  set_io_buffer(fp2, output_buf2, IO_BUFFER_SIZE);

	  fnameout[11]='z';

	  if ((fp3=fopen(fnameout,"wb"))==NULL)
	   {	
			printf("the output file3 can not open\n");
			exit(0);
	   }
	  set_io_buffer(fp3, output_buf3, IO_BUFFER_SIZE);

	  fnameout[9]='h';
	  fnameout[10]='y';
	  fnameout[11]='d';

	  if ((fp4=fopen(fnameout,"wb"))==NULL)
	   {	
			printf("the output file4 can not open\n");
			exit(0);
	   }
	  set_io_buffer(fp4, output_buf4, IO_BUFFER_SIZE);


	  /* check the file length and get the number of data points */

	  fseek(fpin,0L,2);
	  ifl=ftell(fpin);
	  rewind(fpin);

	  sp.npts=ifl/sizeof(data1); 

	  printf("npts=%" SAC_LONG_FMT "\n", sp.npts);

	  /* setup other sac header parameters    */
	 
	  
	  sp.delta = 1./sps;
	  sp.b = 0.0;

	  sp.iftype = ITIME;
	  sp.iztype = IB;

	  sp.leven = TRUE;
	  sp.lcalda = TRUE;

	  strcpy(sp.kstnm,"OBS");

	  /*write binary sac header  */

	  sp.cmpinc = 90.0;
	  strcpy(sp.kcmpnm,"SHX");

	  if(fwrite(&sp,sizeof(struct sac),1,fp1)!=1)
	   {
			printf("FATAL ERROR:bad write%s,header\n",fp1);
			exit(4);
	   }

	  sp.cmpinc = 90.0;
	  strcpy(sp.kcmpnm,"SHY");

	  if(fwrite(&sp,sizeof(struct sac),1,fp2)!=1)
	   {
			printf("FATAL ERROR:bad write%s,header\n",fp2);
			exit(4);
	   }

	  sp.cmpinc = 0.0;
	  sp.cmpaz = 0.0;
	  strcpy(sp.kcmpnm,"SHZ");

	  if(fwrite(&sp,sizeof(struct sac),1,fp3)!=1)
	   {
			printf("FATAL ERROR:bad write%s,header\n",fp3);
			exit(4);
	   }

	  sp.cmpinc = 12345.;
	  sp.cmpaz = 12345.;
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
	  
	  /* Free I/O buffers */
	  if (input_buffer) free(input_buffer);
	  if (output_buf1) free(output_buf1);
	  if (output_buf2) free(output_buf2);
	  if (output_buf3) free(output_buf3);
	  if (output_buf4) free(output_buf4);
  }
 else
  {
	  /* Allocate buffers for 3-channel output files */
	  char *output_buf1_3ch = (char*)malloc(IO_BUFFER_SIZE);
	  char *output_buf2_3ch = (char*)malloc(IO_BUFFER_SIZE);
	  char *output_buf3_3ch = (char*)malloc(IO_BUFFER_SIZE);

	  fnameout[9]='s';
	  fnameout[10]='h';
	  fnameout[11]='x';

	  if ((fp1=fopen(fnameout,"wb"))==NULL)
	   {	printf("the output file1 can not open\n");
			exit(0);
	   }
	  set_io_buffer(fp1, output_buf1_3ch, IO_BUFFER_SIZE);

	  fnameout[11]='y';

	  if ((fp2=fopen(fnameout,"wb"))==NULL)
	   {printf("the output file2 can not open\n");
		exit(0);
	   }
	  set_io_buffer(fp2, output_buf2_3ch, IO_BUFFER_SIZE);

	  fnameout[11]='z';

	  if ((fp3=fopen(fnameout,"wb"))==NULL)
	   {	
			printf("the output file3 can not open\n");
			exit(0);
	   }
	  set_io_buffer(fp3, output_buf3_3ch, IO_BUFFER_SIZE);

	  /* check the file length and get the number of data points */

	  fseek(fpin,0L,2);
	  ifl=ftell(fpin);
	  rewind(fpin);

	  sp.npts=ifl/sizeof(data2); 

	  printf("npts=%" SAC_LONG_FMT "\n", sp.npts);

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
	  
	  /* Free I/O buffers for 3-channel files */
	  if (input_buffer) free(input_buffer);
	  if (output_buf1_3ch) free(output_buf1_3ch);
	  if (output_buf2_3ch) free(output_buf2_3ch);
	  if (output_buf3_3ch) free(output_buf3_3ch);


	  }

}    /* main program end  */


long julian_day(int year, int mon, int day)
{return((long)(day-32075+1461*(year+4800-(14-mon)/12)/4+367*(mon-2+(14-mon)/12*12)/12-3
*((year+4900-(14-mon)/12)/100)/4));}

unsigned int HextoDec(unsigned int a0, unsigned char *p)
{
   unsigned int i,j,sum;
   sum=0;
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
