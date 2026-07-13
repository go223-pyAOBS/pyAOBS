/* Copyright (c) SCSIO, Geoscience department, 2014  */
/* All rights reserved.                              */  
/*----------------------------------------------------------------------
 *
	program name:	sacresamp.c  (sac format data resample in time)
	written by:	Xuelin Qiu     
	@:		Geoscience department, SCSIO, China Academy
                        Tel: 020/89023156   Fax: 020/84451672 
                        e-mail: xlqiu@scsio.ac.cn
	first attempt:	2014/02/18
           Useage: sacresamp  input_sacfile  ndt  output_sacfile        
         Example1: sacresamp  obs03_shz.sac  0.004  obs03_dt4ms.sac
         Example1: sacresamp obs03_shz.sac 3.657E-03 obs03_dt4ms.sac

*******************************************************************************/

   #include "stdio.h"         /* Standard I/O header file  */
   #include "stdlib.h"
   #include "sac_new.h"
   #include "math.h"

   struct sac sp;
 
   main(argc,argv)
   int argc;
   char *argv[];
  {  
   FILE *Sacinf;         /*pointer to the input file */
   FILE *Sacoutf;         /*pointer to the output file */
   long i,j,snpts,nnpts; 
   float sdt,ndt;
   double stlen,ntlen;
   float sdata1,sdata2,ndata;
   double ti,tj,tij;
 
   if (argc<4)
      {printf("you forgot a filename or the ndt\n");
       printf("Usage: sacresamp input_sacfile ndt output_sacfile\n");
       printf("Example: sacresamp obs03_shz.sac 0.004 obs03_dt4ms.sac\n");
       exit(0);
      } 
/* open the input file   */
  if ((Sacinf=fopen(argv[1],"ra"))==NULL)
   if ((Sacinf=fopen(argv[1],"ra"))==NULL)
      {printf("the SAC input file can not open\n");
       exit(0);
      }

/* obtain the new sampling interval   */
    sscanf(argv[2],"%f",&ndt);
    printf("ndt= %f (ms)\n",ndt*1000);

/* open the output file   */
  if((Sacoutf=fopen(argv[3],"wb"))==NULL)
    {printf("cannot open the outfile1\n");
     exit(0);
    }

/* first read the sac header (158 words i.e. 632 bytes) */

   fread(&sp,sizeof(struct sac),1,Sacinf);

      printf("sac dt= %f (ms)\n",sp.delta*1000);
      if (ndt==sp.delta)
      {printf("The new dt is the same as the input sac dt\n");
       exit(0);
      } 

      sdt=sp.delta;
      snpts=sp.npts;

      stlen=sdt*(snpts-1);  // sac time length
 
      nnpts=(long int)(stlen/ndt); // new number of points

      ntlen=ndt*(nnpts-1);  // new time length

      if (ntlen>stlen)
      {printf("New time length is larger than old length\n");
       exit(0);
      } 

/* write the new sac header (158 words i.e. 632 bytes) */

      sp.delta=ndt;
      sp.npts=nnpts;
      sp.e=sp.b+ntlen;

   fwrite(&sp,sizeof(struct sac),1,Sacoutf);

/*  begin read and writing floating point data
    new data points are calculated by linear interpolation 
    of the two nearby data points just before and after */

    i=1;j=1;

    while(i<=snpts)
    {
    fread(&sdata1,4,1,Sacinf); 
    ti=(i-1)*sdt;

       while(j<=nnpts)
       {
       tj=(j-1)*ndt;
       if(tj>ti) break;
       if(tj==ti) ndata=sdata1;
       else {
            tij=ti-tj;    // linear interporated
            ndata=sdata1-(sdata1-sdata2)*tij/sdt;  
            }
       fwrite(&ndata,4,1,Sacoutf);
       j++;
       }

    sdata2=sdata1;
    i++;
    }

   fclose(Sacinf);
   fclose(Sacoutf);

   }                      /* main program end */



