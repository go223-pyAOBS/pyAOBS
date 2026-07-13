/* Copyright (c) SCSIO, Geoscience department, 2005/10 */
/* All rights reserved.                              */  
/*----------------------------------------------------------------------
 *
	program name:	sac2y.c  (sac format convert to segy format)
	written by:	Minghui Zhao & Xuelin Qiu     
	@:		Geoscience department, SCSIO, China Academy
                        Tel: 020/89023192   Fax: 020/84451672 
                        e-mail: mhzhao@scsio.ac.cn
	    update:	05/10/07 compile by gnu free software
        updated by:     Shaohong Xia
        modified by:   Xuelin Qiu  22/09/08                       
        Version:        Ver.1.1
                        merge the 1st trace and the 2nd+ traces
                        finish when SAC file ends
       NOTE: this program add a parameter "deltr" in the file "par.in". tr.deltr=tcoor*(-1000), by Minghui Zhao.

		updated by Aowei 2010/03/19, adding the drift time of OBS. see line99, and 147-152,361-362.

		updated by Haoyu Zhang 2017/04/15 as "v2.1" version to be applicable for non-integral sampling rate, in line 61,67,112-116,211-214,287,291,407-408,415-417,446-449,458,etc.
		p.s. Some dispensable parameters are added to do some tests.


*******************************************************************************/

   #include "stdio.h"         /* Standard I/O header file  */
   #include "sac_new.h"
   #include "stdlib.h"
   #include "math.h"
   #include "segy.h"
   #include "gmt_proj.h"    /* Define project_info structures */
   #define SEEK_BEG 0
/* subroutine prototypes */
   int vtm ();             /*      Initialize TM projection        */
   int utm();		/*	Convert lon/lat to x/y (UTM)	*/

/*subprograms prototype */

   static void float_to_ibm(int from[], int to[], int n, int endian);
   unsigned long up_side_down4(unsigned long x);
   short int up_side_down2(short int x);
   long julian_day(long year, long mon, long day);

   segy tr;
   bhed bh;
   struct sac sp;
   struct MAP_PROJECTIONS project_info;
   double map_scale_factor;
   double eq_radius, pol_radius, rf;
   double EQ_RAD, f, ECC2, ECC4, ECC6, ECC;


   main(argc,argv)
   int argc;
   char *argv[];
  {  
   FILE *Sf,*Uf,*Pf,*fp,*dp,*lp;      /*pointer to the input file */
   FILE *Yf,*testTT;             /*pointer to the segy-format output file */
   int i=0,j=0,length,m,l,ns1,n,k,nn,lat3,lon3; 
   int endday,styear,stmonth,stday,sthour,stmin,stsec,stmsec,edyear,edmonth,edday,edhour,edmin,edsec,edmsec,stjday,edjday,sacday;
   long int t4,TT,T,TT0,TT1,TT0_temp,t4_temp,TT_temp,bias;
   double time,t2,dec,dis,dis1;
   double x1_out,y1_out,x2_out, y2_out;
   double lat1,lon1,lon,lat,lon2,lat2,lon1_in,lat1_in,lon0;
   char str2[10],c,c1,c2,p[2];
   static   char   ebcbuf[3200];     /* ebcdic data buffer */
   char blank[400];
   float del,drift,drift1,drift2,drift3;
   double t3,t1,t6,tt,timegap;
   float d,v,h,ex,ey,et;
   int iv,dv,tcoor,samv,samn,samv_temp,samn_temp;
   float data[500000],arr[500000];

   int hour,min,sec,msec;
   int latdeg,latmin,londeg,lonmin;
   int sec1,sec2,sec3,sec4;
   double latsec,lonsec;
   int water_depth;
   char latNS,longEW;
   char line[100], subs[100],str1[2];
   int get_subs(char*,char*,int,int);
   double distance(double lat,double lon,double slat,double slon);
   float rv;
   long int datPtLoc,datPtLoc_temp;
   short trNum;
   
 
   if (argc!=4)
      {printf("you forgot to enter a filename\n");
       exit(0);
      }
/* open the input file   */
   if ((Sf=fopen(argv[1],"ra"))==NULL)
      {printf("the SAC file can not open\n");
       exit(0);
      }
   if ((Uf=fopen(argv[2],"ra"))==NULL)
      {printf("the UKOOA file can not open\n");
       exit(0);
      }
 /* open the output file */
   if((Yf=fopen(argv[3],"wb"))==NULL)
     { printf("Cannot open the SEGY output file\n");
       exit(0) ;
     }
   
  /* open the parameter file*/
   if((Pf=fopen("par.in","ra"))==NULL)
     { printf("Cannot open the parameter file: par.in\n");
       exit(0);
     } 
	 //create a file named "sac2yBias.txt" 
	 //containing the difference of 
	 //addressing for beginning of each trace 
	 //between v2-1 and its former verison 
	//testTT=fopen("sac2yBias.txt","w");
	
   while ( fgets(line, 100, Pf) != NULL ) 
   {
        if ( line[0] == 'P'&&line[1]=='4' )
          {fscanf(Pf,"%f,%d,%d,%lf,%f,%f,%f",&rv,&length,&tcoor,&lon0,&ex,&ey,&et);
           break;
          } 
   }

/* Parameters assigned in map_setup */

/* ELLIPSOID		= WGS-84 */

   eq_radius  = 6378137.0;
   pol_radius = 6356752.3;
   rf = 298.257223563; /* GMT3.3.4 and HYPACK */

   EQ_RAD = eq_radius;
   f = 1.0/rf;

   ECC2 = 2 * f - f * f;
   ECC4 = ECC2 * ECC2;
   ECC6 = ECC2 * ECC4;
   ECC  = d_sqrt (ECC2);

/* Parameters assigned in gmt_init.h and gmt_defaults.h */

   map_scale_factor = 0.9996;  /* gmtdefs.map_scale_factor */
   
   vtm (lon0);       /* Set up an UTM or TM projection */
  
   /* first read the  sac header   (158 bytes)   */

   fread(&sp,sizeof(struct sac),1,Sf);
 
   if (sp.b != 0.0)
      { printf("the sp.b= %lf\n",sp.b);
        printf ("The begining time of SAC file is not zero/n");
        exit (0);
      }

   t1=sp.nzhour*3600+sp.nzmin*60+sp.nzsec+(float)(sp.nzmsec)/1000;
   printf("%d  %d  %d  %d  %lf\n",sp.nzjday,sp.nzhour,sp.nzmin,sp.nzsec,t1);

/* add the OBS drift correction and recording delay. 
the drift corresponding to the whole recording file, start recording time and end recording time */         

 dp=fopen("drift.in","r");
 fscanf (dp,"%d %d %d %d %d %d %d %d %d %d %d %d %f\n",&sthour,&stmin,&stsec,&styear,&stmonth,&stday,&edhour,&edmin,&edsec,&edyear,&edmonth,&edday,&drift);  

 printf("sthour,stmin are %d %d\n",sthour,stmin);
 edmsec=0;
 stmsec=0;
 
  stjday=julian_day((long)styear,stmonth,stday);
  edjday=julian_day((long)edyear,edmonth,edday);
  sacday=julian_day((long)styear,1,1);
  stjday=(short)(stjday-sacday+1);
  edjday=(short)(edjday-sacday+1);   
  printf("stjday edjday are %d %d\n", stjday, edjday);

  printf("drift=%fms\n",drift);
  timegap=(edjday-stjday+1)*24*3600+(edhour-sthour)*3600+(edmin-stmin)*60+(edsec-stsec)+(edmsec-stmsec)/1000;
  printf("timegap=%lf\n",timegap);
  drift1=(double)drift/timegap;
  printf("Drift Rate(ms/s):%lf\n",drift1);

  if (sp.stla==-12345)    
  {lp=fopen("loc.in","r");
  fscanf(lp,"%lf %lf\n",&lon1,&lat1);
 /* printf("lat=%lf lon=%lf\n",lat1,lon1);*/}
  else 
 { lat1=(double)sp.stla;  lon1=(double)sp.stlo;
  printf(" the corrected position are %lf %lf\n",lat1,lon1) ;}
  /* printf("%10.3lf %10.3lf\n",lon1,lat1);*/

   project_info.north_pole = 1; /* TRUE if projection is on northern hermisphere */
   if (lat1<0) {
   project_info.north_pole = 0;} /* southern hermisphere */

   utm(lon1,lat1,&x1_out,&y1_out);  
    
  /*printf("%10.3lf %10.3lf\n",x1_out,y1_out);*/

 
  /* convert sac.h parameters to  bhed and segy header parameters    */

  /* setup binary header 3201-3600 bytes */
   printf("delta in SAC inputed:%fs\n",sp.delta);
   //bh.ntrpr =  length*(int)(1/sp.delta+0.5);        /* number of data traces */
   //bh.hdt  =  (float)(1000000*sp.delta+0.5) ; /*sample interval in micro secs for this 
     //                                         reel 1000000*sp.delta */
   bh.hdt  =  (int)(1000000*sp.delta+0.5) ; /*sample interval in micro secs for this 
                                              reel 1000000*sp.delta */
   printf("Sampling delta:%d microsecond\n",bh.hdt);
      
   //bh.hns   =  length *  (int)(1/sp.delta+0.5);  /*number of samples per trace for this reel*/
   bh.hns   =  (int)(((double)length)/sp.delta+0.5);  /*number of samples per trace for this reel*/
   printf("Samples per trace:%d\n",bh.hns);
   bh.format = 1;         /* data sample format code:
				1 = floating point (4 bytes)
				2 = fixed point (4 bytes)
				3 = fixed point (2 bytes) 
                        	4 = fixed point w/gain code (4 bytes) */
      
   bh.fold =   1;      /* CDP fold expected per CDP ensemble */
        
   bh.mfeet =  1;             /* measurement system code: 1 = meters, 2 = feet */

/* read the trace identification header (240 bytes) */

   tr.tracf = 1;            /* field record number */
   tr.dt   = bh.hdt ;       /* sample interval; in micro-seconds */       
   tr.ns   =   bh.hns;      /* number of samples in this trace =bh.hdt */    
   //printf("%d\n",tr.ns);   
   tr.trid =   1;           /* tracd number within CDP ensemble */
   tr.nvs = 1;              /* number of vertically summed traces */
   tr.nhs = 1;              /* number of horizontally summed traces */
   tr.duse = 1;             /* data use:
                                1=production   2=test       */
   tr.gelev = sp.evel;         /* receiver group elevation from sea level */
   tr.scalel = 0;       
   tr.scalco = -1;
   tr.counit = 1;           /*coordinate unit code for tr.gx,tr.gy,tr.sx and tr.sy:
                                       1=length(meters or feet)
                                       2=second of arc           */

   tr.delrt = tcoor*(-1000);
       
 /* write ebcdic block and binary header  */

   //bh.fold =   up_side_down2(bh.fold);
   bh.format =  up_side_down2(bh.format); 
   bh.hns   =  up_side_down2(bh.hns);
   bh.hdt  =   up_side_down2(bh.hdt);      
   //bh.ntrpr =  up_side_down2(bh.ntrpr);
   bh.mfeet =  up_side_down2(bh.mfeet);

   fwrite(&ebcbuf,3200,1,Yf); 
   //fwrite(&bh,400,1,Yf); 
   //set room for binary file Header and skip it.
   fwrite(&blank,400,1,Yf); 
           
          
/* write the trace header and data */

   tr.tracf = up_side_down4(tr.tracf);
   tr.dt    = up_side_down2(tr.dt);
   tr.ns    = up_side_down2(tr.ns);
   tr.nvs   = up_side_down2(tr.nvs);
   tr.nhs   = up_side_down2(tr.nhs);
   tr.duse  = up_side_down2(tr.duse);

   tr.delrt = up_side_down2(tr.delrt);
 
   tr.trid  = up_side_down2(tr.trid);
   tr.gelev = up_side_down4(tr.gelev);
   tr.scalel= up_side_down2(tr.scalel);
   tr.scalco= up_side_down2(tr.scalco);
   tr.counit= up_side_down2(tr.counit);

   tr.gx = (int)x1_out;
   tr.gy = (int)y1_out;
   tr.gx  =  up_side_down4(tr.gx);
   tr.gy  =  up_side_down4(tr.gy);

   samv=(int)(1/sp.delta+0.5);
   
   //samn_temp for samn
   samn=length*samv;
   samn_temp=(int)(((double)length)/sp.delta+0.5);
   
   //TT0_temp for TT0
   TT0 = -samn;
   TT0_temp = -samn_temp;
/*  search UKOOA file to find the first shot point */
	trNum = 0;

   while ( fgets(line, 85, Uf) != NULL ) 
   {
        if ( line[0] == 'S' || line[0] == 'N' ) 
        {  //tr_count += 1;
			//printf("%d\n",tr_count);
           get_subs(subs, line, 20, 24); 
           n = (int)(atoi(subs));
           /*printf("%d\n",n);*/
           get_subs(subs, line, 5, 7); 
           tr.day = atoi(subs); 
           //printf("%d\n",tr.day);

           if(tr.day>=sp.nzjday)
           {          
              tr.year=sp.nzyear;
 
              get_subs(subs, line, 5, 7); 
              tr.day = atoi(subs);                      
              get_subs(subs, line, 8, 9); 
              tr.hour    = atoi(subs); 
              get_subs(subs, line, 10, 11); 
              tr.minute    = atoi(subs); 
              get_subs(subs, line, 12, 13); 
              tr.sec    = atoi(subs); 
              get_subs(subs, line, 15, 17); 
              tr.timbas   = atoi(subs); 

              get_subs(subs, line, 26, 27); 
              latdeg    = atoi(subs); 
              get_subs(subs, line, 28, 29); 
              latmin    = atoi(subs);
              get_subs(subs, line, 30, 31); 
              sec1    = atoi(subs);
              get_subs(subs, line, 33, 34); 
              sec2    = atoi(subs);
              latsec=sec1+(double)(sec2)/100;
              lat2=latdeg+(double)(latmin)/60+(double)(latsec)/3600;

              if (line[35]=='S') lat2=-lat2; // southern hemisphere

              get_subs(subs, line, 36, 38); 
              londeg    = atoi(subs); 
              get_subs(subs, line, 39, 40); 
              lonmin    = atoi(subs); 
              get_subs(subs, line, 41, 42); 
              sec3    = atoi(subs);
              get_subs(subs, line, 44, 45); 
              sec4    = atoi(subs);
              lonsec=sec3+(double)(sec4)/100;
              lon2=londeg+(double)(lonmin)/60+(double)(lonsec)/3600;                    
 
              get_subs(subs, line, 65,68);
              sec4  =atoi(subs);
              tr.swdep  = sec4;

              get_subs(subs, line, 72,75); 
              tr.fldr = atoi(subs);               /* field record number  */           
     
  
 /* add to water depth,CDP number and x&y coordinate for each trace */
 
              
              tr.ep = n;                 /* energy source point number */
              tr.cdp = n;                /* CDP ensemble number  */    
    
              utm(lon2,lat2,&x2_out,&y2_out);

            /*the distance of antenna to airgun is 86.0 m, have been corrected in UKOOA file,so make ex and ey equal to zero in par.in */
           /*the distance of antenna to airgun is 37.8m , ex=37.8*sin12.5=8.1814  ey=37.8*cos12.5=36.9040 */

              x2_out=x2_out+ex;
              y2_out=y2_out+ey;
  
   
              tr.sx=(int)x2_out;
              tr.sy=(int)y2_out;
              /*printf("%lf,%lf\n",x2_out,x1_out);*/
              dis=sqrt(pow((x2_out-x1_out),2)+pow((y2_out-y1_out),2));
              tr.offset = (int)(dis+0.5);   /* caculate the distance from     
                                            source point to receiver group */

 /*add to the begaining time for each trace */
    
              if (rv>0.001)     t6=(double)(tr.offset)/(rv*1000.);
              if (rv>100.0 || rv<=0.001) t6=0.;

              drift2=(tr.day-stjday)*24*3600+(tr.hour-sthour)*3600+(tr.minute-stmin)*60+(tr.sec-stsec)+(double)(tr.timbas-stmsec)/1000;
              drift3=drift2*drift1;

 /*** t3 is the original time of current shot plus reduced time ***/

              t3=tr.hour*3600+tr.minute*60+tr.sec+(double)(tr.timbas)/1000+t6+et+drift3/1000;
           
              tr.hour=(int)(t3/3600);       
              tr.minute=(int)((t3-tr.hour*3600)/60);
              tr.sec=(int)(t3-tr.hour*3600-tr.minute*60);
              tr.timbas=(int)(1000*(t3-tr.hour*3600-tr.minute*60-tr.sec)+0.5);  
              /* printf("%lf\n",t3);*/
   
              if(lat2<=lat1) tr.offset=-tr.offset;  /*lat1 or lon1 for OBS, lat2 or lon2 for shot location*/
             /* t3=t3+86400*(tr.day-sp.nzjday);*/
              if(tr.day>sp.nzjday) t3=t3+24*3600*(tr.day-sp.nzjday); 
         
               
  /* firstly find the first point of the 1st trace data */
				//t4_temp for t4
				t4=(int)(samv*(t3-t1)+0.5)-samv*tcoor;
                t4_temp=(int)((1.0/sp.delta)*(t3-t1)-(1.0/sp.delta)*tcoor+0.5);
				//printf("t4:%ld	t4_temp:%ld\n",t4,t4_temp);
				
                 if (t4_temp<0) continue;    
				 
				/* next shot if sac data not start yet */
                 if (t4_temp+samn_temp>sp.npts) break;   /* finish if sac ends */    
				
				//TT_temp for TT
                 TT=t4-TT0-samn;
				 TT_temp=t4_temp-TT0_temp-samn_temp;
				 //printf("%ld %ld\n",t4_temp, TT_temp);
                 if (TT_temp<0) continue;       /* next shot if too close to previous shot */
				
			     //printf("%d\n",tr_count);
                 tr.tracl = 1 + i++;      /* trace sequence number within line */
                 tr.tracr = 1 + j++;      /* trace sequence number within reel */
				 //bias = t4-t4_temp;
			     //fprintf(testTT,"%ld %ld %ld\n",t4,t4_temp,bias);
                 fseek(Sf,4*TT_temp,SEEK_CUR);
                 fread(data,4,samn_temp,Sf); 
                          
    /*convert */          
                 tr.tracl =  up_side_down4(tr.tracl);
                 tr.tracr =  up_side_down4(tr.tracr);
                 tr.swdep = up_side_down4(tr.swdep);
                 tr.fldr =  up_side_down4(tr.fldr);
                 tr.ep =  up_side_down4(tr.ep);
                 tr.cdp =  up_side_down4(tr.cdp);

                 tr.year = up_side_down2(tr.year);
                 tr.day  = up_side_down2(tr.day);
                 tr.hour = up_side_down2(tr.hour);
                 tr.minute  = up_side_down2(tr.minute);
                 tr.sec  = up_side_down2(tr.sec);
                 tr.timbas  = up_side_down2(tr.timbas);
                 tr.sx = up_side_down4(tr.sx);
                 tr.sy = up_side_down4(tr.sy);
              
                 tr.offset= up_side_down4(tr.offset);
      
                 /* Convert internal floats to IBM floats */

	         float_to_ibm((int *) data, (int *) 
				data,samn_temp, 0);
                               
                 fwrite(&tr,240,1,Yf);       /* write the 240-byte trace header  */ 

                 fwrite(data,4,samn_temp,Yf);     /* write the first trace data */
                 trNum += 1;   
				
                 TT0=t4;
				 TT0_temp=t4_temp;

                
         
           }       
         }
     }
	 
	 //rewind to write binary file header
	 fseek(Yf,3200,0);
	 bh.ntrpr = trNum;    /* number of data traces */
	 printf("Number of traces:%d\n",bh.ntrpr);
	 bh.ntrpr =  up_side_down2(bh.ntrpr);
	 bh.fold =   bh.ntrpr;
	 bh.fold =   up_side_down2(bh.fold);
	 fwrite(&bh,400,1,Yf); 
	 fseek(Yf,0,2);
     fclose(Sf);
     fclose(Yf);
     fclose(Uf);
	 //fclose(testTT);
	 printf("**********************************************************\n");
   }                      /* main program end */




   /* Assumes sizeof(int) == 4 */
   static void float_to_ibm(int from[], int to[], int n, int endian)
/**********************************************************************
   float_to_ibm - convert between 32 bit IBM and IEEE floating numbers
*********************************************************************** 
   Input:
   from	   input vector
   n	   number of floats in vectors
   endian  =0 for little endian machine, =1 for big endian machines

   Output:
   to	   output vector, can be same as input vector

*********************************************************************** 
   Notes:
   Up to 3 bits lost on IEEE -> IBM

   IBM -> IEEE may overflow or underflow, taken care of by 
   substituting large number or zero

   Only integer shifting and masking are used.
*********************************************************************** 
   Credits:     CWP: Brian Sumner
***********************************************************************/
   {
    register int fconv, fmant, i, t;

    for (i=0;i<n;++i) 
    {
     fconv = from[i];
     if (fconv) 
     {
      fmant = (0x007fffff & fconv) | 0x00800000;
      t = (int) ((0x7f800000 & fconv) >> 23) - 126;
      while (t & 0x3) { ++t; fmant >>= 1; }
      fconv = (0x80000000 & fconv) | (((t>>2) + 64) << 24) | fmant;
     }
     if(endian==0) fconv = (fconv<<24) | ((fconv>>24)&0xff) |
                           ((fconv&0xff00)<<8) | ((fconv&0xff0000)>>8);

     to[i] = fconv;
    }
    return;
   }

   unsigned long up_side_down4(unsigned long x)
   { 
    x=((x<<24)&0xff000000)|((x<<8)&0x00ff0000)|((x>>24)&0x000000ff)
		|((x>>8)&0x0000ff00);
    return x;  
   }

   short int  up_side_down2(short int x)
   { 
    x=((x>>8)&0x00ff)|((x<<8)&0xff00);
    return x;  
   }



/*************************************************************************
       function converting lon/lat to UTM x/y
***************************************************************************/

/*TRANSFORMATION ROUTINES FOR THE UNIVERSAL TRANSVERSE MERCATOR PROJECTION (UTM)*/

   int utm (lon, lat, x, y)
   double lon, lat, *x, *y; 
   {

   /* Convert lon/lat to UTM x/y */

   if (lon < 0.0) lon += 360.0;
   tm (lon, lat, x, y);
   (*x) += 500000.0;
   if (!project_info.north_pole) (*y) += 10000000.0;  /*For S hemisphere,add 10^6 m*/
   }

   int iutm (lon, lat, x, y)
   double *lon, *lat, x, y; 
   {
   /* Convert UTM x/y to lon/lat */

   x -= 500000.0;
   if (!project_info.north_pole) y -= 10000000.0;
   itm (lon, lat, x, y);
   }


/*TRANSFORMATION ROUTINES FOR THE TRANSVERSE MERCATOR PROJECTION (TM)*/

   int vtm (lon0)
   double lon0; 
   {
   /* Set up an TM projection */
   double e1;
	
   e1 = (1.0 - d_sqrt (1.0 - ECC2)) / (1.0 + d_sqrt (1.0 - ECC2));
   project_info.t_e2 = ECC2 / (1.0 - ECC2);
   project_info.t_c1 = (1.0 - 0.25 * ECC2 - 3.0 * ECC4 / 64.0 - 5.0 * ECC6 / 256.0);
   project_info.t_c2 = (3.0 * ECC2 / 8.0 + 3.0 * ECC4 / 32.0 + 45.0 * ECC6 / 1024.0);
   project_info.t_c3 = (15.0 * ECC4 / 256.0 + 45.0 * ECC6 / 1024.0);
   project_info.t_c4 = (35.0 * ECC6 / 3072.0);
   project_info.t_ic1 = (1.5 * e1 - 27.0 * pow (e1, 3.0) / 32.0);
   project_info.t_ic2 = (21.0 * e1 * e1 / 16.0 - 55.0 * pow (e1, 4.0) / 32.0);
   project_info.t_ic3 = (151.0 * pow (e1, 3.0) / 96.0);
   project_info.t_ic4 = (1097.0 * pow (e1, 4.0) / 512.0);
   project_info.central_meridian = lon0;
   }

   int tm (lon, lat, x, y)
   double lon, lat, *x, *y; 
   {
   /* Convert lon/lat to TM x/y */
   double N, T, T2, C, A, M, dlon, tan_lat, cos_lat, A2, A3, A5;
	
   dlon = lon - project_info.central_meridian;
   if (fabs (dlon) > 360.0) dlon += copysign (360.0, -dlon);
   if (fabs (dlon) > 180.0) dlon = copysign (360.0 - fabs (dlon), -dlon);
   lat *= D2R;
   M = EQ_RAD * (project_info.t_c1 * lat - project_info.t_c2 * sin (2.0 * lat)
       + project_info.t_c3 * sin (4.0 * lat) - project_info.t_c4 * sin(6.0  * lat));
   if (fabs (lat) == M_PI_2) 
    {
    *x = 0.0;
    *y = map_scale_factor * M; /* gmtdefs.map_scale_factor */
    }
	
    else 
    {
    N = EQ_RAD / d_sqrt (1.0 - ECC2 * pow (sin (lat), 2.0));
    tan_lat = tan (lat);
    cos_lat = cos (lat);
    T = tan_lat * tan_lat;
    T2 = T * T;
    C = project_info.t_e2 * cos_lat * cos_lat;
    A = dlon * D2R * cos_lat;
    A2 = A * A;	A3 = A2 * A;	A5 = A3 * A2;
    *x = map_scale_factor * N * (A + (1.0 - T + C) * (A3 * 0.16666666666666666667)
			+ (5.0 - 18.0 * T + T2 + 72.0 * C - 58.0 * project_info.t_e2) * (A5 * 0.00833333333333333333));
    A3 *= A;	
    A5 *= A;
    *y = map_scale_factor * (M + N * tan (lat) * (0.5 * A2 + (5.0 - T + 9.0 * C + 4.0 * C * C) * (A3 * 0.04166666666666666667)
			+ (61.0 - 58.0 * T + T2 + 600.0 * C - 330.0 * project_info.t_e2) * (A5 * 0.00138888888888888889)));
    }
   }

int itm (lon, lat, x, y)
double *lon, *lat, x, y; {
	/* Convert TM x/y to lon/lat */
	double M, mu, phi1, C1, C12, T1, T12, tmp, tmp2, N1, R1, D, D2, D3, D5, cos_phi1, tan_phi1;
	
	M = y / map_scale_factor;  /* gmtdefs.map_scale_factor */
	mu = M / (EQ_RAD * project_info.t_c1);
	phi1 = mu + project_info.t_ic1 * sin (2.0 * mu) + project_info.t_ic2 * sin (4.0 * mu)
		+ project_info.t_ic3 * sin (6.0 * mu) + project_info.t_ic4 * sin (8.0 * mu);
	cos_phi1 = cos (phi1);
	tan_phi1 = tan (phi1);
	C1 = project_info.t_e2 * cos_phi1 * cos_phi1;
	C12 = C1 * C1;
	T1 = tan_phi1 * tan_phi1;
	T12 = T1 * T1;
	tmp = 1.0 - ECC2 * (1.0 - cos_phi1 * cos_phi1);
	tmp2 = d_sqrt (tmp);
	N1 = EQ_RAD / tmp2;
	R1 = EQ_RAD * (1.0 - ECC2) / (tmp * tmp2);
	D = x / (N1 * map_scale_factor);   /* gmtdefs.map_scale_factor */
	D2 = D * D;	D3 = D2 * D;	D5 = D3 * D2;
	
	*lon = project_info.central_meridian + R2D * (D - (1.0 + 2.0 * T1 + C1) * (D3 * 0.16666666666666666667) 
		+ (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C12 + 8.0 * project_info.t_e2 + 24.0 * T12)
		* (D5 * 0.00833333333333333333)) / cos (phi1);
	D3 *= D;	D5 *= D;
	*lat = phi1 - (N1 * tan (phi1) / R1) * (0.5 * D2 -
		(5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C12 - 9.0 * project_info.t_e2) * (D3 * 0.04166666666666666667)
		+ (61.0 + 90.0 * T1 + 298 * C1 + 45.0 * T12 - 252.0 * project_info.t_e2 - 3.0* C12) * (D5 * 0.00138888888888888889));
	(*lat) *= R2D;
}


/*******************************************************************
  function to extract a substring from begin-th char to end-th char
    return number of non-blank characters 
********************************************************************/

int get_subs(char substring[], char line[], int begin, int end)
{ 
    short  i, j=0;

    for (i = begin-1; i < end; ++i) {
        if (line[i] != ' ') {
            substring[j] = line[i];
            ++j;
        }
    }
    substring[j] = '\0';
    return j;
}

long julian_day(long year, long mon, long day)
{return(day-32075+1461*(year+4800-(14-mon)/12)/4+367*(mon-2+(14-mon)/12*12)/12-3*((year+4900-(14-mon)/12)/100)/4);}
