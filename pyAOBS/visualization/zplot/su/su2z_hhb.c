/* Copyright (c) GEOMAR, 1997.*/
/* All rights reserved.                       */

/* SUSETHW: $Revision: 1.0 $ ; $Date: 1997/07/18  $		*/
/* modify by Haibo  20230828 */

#include "su.h"
#include "segy.h"
#include "header.h"

#define MAXPICK 40
#define HROT_TVERT TVERT
#define HROT_TINLIN TINLIN
#define HROT_TXLIN TXLIN
#define HROT_ROTVERT ROTVERT
#define HROT_TRADIAL TRADIAL
#define HROT_TTRANS TTRANS
#define HROT_HYDRO 33
/*********************** self documentation **********************/
char *sdoc[] = {
"									",
" SU2Z - Convert SU segy data into z format used by Colin Zelt's	",
"        traveltime picking program ZPLOT				",
"        CDP is used as record number   				",
"									",
"  su2z <stdin >stdout [optional parameters]				",
"									",
" Optional parameters:							",
"									",
"  zdata=data.z    output z-data file 					",
"  hfile=file.hdr  output header file	 				",
"									",
"  key1=fldr       profile number					    ",
"  key2=cdp        record (gather) number				",
"  key3=tracf      trace number	within record           ",
"  key4=trid       component name                       ",
"                      raw  12-vertical,13-crossline, 14-inline",
"		           rotated  15-vertical,16-transverse,17-radial     ",
"                           11-hydrophone    ",
"  nrec=0          number of records (shot or receiver gathers)		",
"                  =0 determined automatically by counting number	",
"                     of different key2          			",
"									",
"  npick=20        number of picks/phases (max. 20) for a single trace	",
"									",
"  vred=tr.swevel/1000.0						",
"                  reduction velocity (km/s)				",
"									",
"  tstart=tr.stas							",
"                  trace start time relative to reduced time in msec	",
"									",
"  tend=tr.stae								",
"                  trace end time relative to reduced time in msec	",
"									",
"  verbose=0       silent operation                                	",
"                  =1 ; echo header and record info           		",
"                  =2 ; echo additionally trace info           		",
"									",
"									",
" Examples:								",
"    ... | su2z npick=10 vred=6.0 tstart=-5000 tend=10000 \\			",
"         zdata=zdata < in_segy						",
"									",
"									",
NULL};

/* Credits:
 *	GEOMAR: Sanyu Ye
 *  
 */
/**************** end self doc ***********************************/

/* prototype for function used internally */

segy tr;

struct {
	  int	ntraces;  /* total number of traces in file */
	  int	npts;     /* no of points per trace */
	  int	sint;     /* sampling interval (microseconds) */
	  int   tstart;   /* trace start time (ms) */
	  int   tend;     /* trace end time (ms) */
	  int	nrec;     /* no of records (gathers) */
	  int	npick;    /* no of max. picks */
	float	vredf;    /* reduction velocity in km/s */
	  int	ifmt;     /* data format: float=1; short=0 */
	float	xlatlong;
	float	xelev;
	float	xutm;
	float	cm;	      /* central meridian in degree */
	float	dummy[9];
} fh;

struct {
	  int	nrec;	  /* record number (shot station or number) */
	  int	itsn;	  /* trace sequential number within record */
	  int	ireci;	  /* trace number (receiver station or number) */
	  int	itype;	  /* data type: 1=vertical, 2=radial,
					     3=transverse, 4=hydrophone */
	  int	iflag;	  /* data flag 1=alive, other=dead trace */
	float	offset;	  /* receiver-source offset in meters */
	float	azi;	  /* receiver to source azimute (minutes) */
	  int	igain;	  /* multiplicative gain factor */
	float	texact;	  /* difference between first sample and exact start time */
	float	slat;	  /* source latitude (degrees) */
	float	slong;	  /* source longitude */
	  int	selev;	  /* source elevation (meters) */
	  int   swdepth;  /* water depth at source */
	float	rlat;	  /* receiver latitude (degrees) */
	float	rlong;	  /* receiver longitude */
	  int	relev;	  /* receiver elevation (meters) */
	float	sxutm;	  /* source x coordinate in UTM system (m) */
	float	syutm;	  /* source y coordinate in UTM system (m) */
	float	sz;	      /* source z coordinate in UTM system (m) */
	float	rxutm;	  /* receiver x coordinate in UTM system (m) */
	float	ryutm;	  /* receiver y coordinate in UTM system (m) */
	float	rz;	      /* receiver z coordinate in UTM system (m) */
} th;

main(int argc, char **argv)
{
        cwp_String      FileOut, FileHdr;
        FILE           *fpout, *fphdr;  /* Pointer to the output file  */

        cwp_String      key1, key2, key3, key4;
        cwp_String      type1, type2, type3, type4;
	Value	val1, val2, val3, val4;
	int		ival1, ival2, ival3, ival4;
	int		index1, index2, index3, index4;
	float	picks[MAXPICK];		/* values of pick (dummy)	*/
        int 	verbose, i;            	/* echo every ...               */
	int	nbytes;			/* number of bytes for trace header */
	int     recno;			/* current rec no */
	int     nrec;			/* number of records (gathers) */
	int     ntrace;			/* number of trace within record (gather) */

	/* Initialize */
	initargs(argc, argv);
	requestdoc(1);

	/* init headers to null */
	memset((void*)&fh, 0, sizeof(fh));
	memset((void*)&th, 0, sizeof(th));

	fh.ifmt=1;  	/* set data format float (real*4) */
	fh.xlatlong=fh.xelev=fh.xutm=1.0; /* set multiplicative factor */
	th.iflag=1;
	th.igain=0;

    /* data type  determined by the value of key4,           	
       1=vertical, 2=radial, 3=transverse, 4=hydrophone			
	   if th.itype not set, default data type is hydrophone   */
/*	if (!getparint("itype",  &th.itype))	th.itype=4;	*/
    th.itype=4;	
 	if (!getparint("nrec",  &fh.nrec)) 	fh.nrec=0;
 	if (!getparint("verbose", &verbose))	verbose  = 0;

	/* set number of shot records (profiles) */
 	if (!getparstring("key1",  &key1)) 	key1="fldr";
	if (!getparstring("key2",  &key2)) 	key2="cdp";
 	if (!getparstring("key3",  &key3)) 	key3="tracf";
 	if (!getparstring("key4",  &key4)) 	key4="trid";
	type1 = hdtype(key1);
	type2 = hdtype(key2);
	type3 = hdtype(key3);
	type4 = hdtype(key4);	
	index1 = getindex(key1);
	index2 = getindex(key2);
	index3 = getindex(key3);
	index4 = getindex(key4);
	

	/* open output files      */

    	if (!getparstring("zdata"  , &FileOut))		FileOut="data.z";
    	if (!getparstring("hfile"  , &FileHdr))		FileHdr="file.hdr";
        if ((fpout = fopen(FileOut, "w+b")) == NULL) {
            err("\n !!! error opening output z-data file %s !!!\n", FileOut);
        }
        if ((fphdr = fopen(FileHdr, "w+b")) == NULL) {
            err("\n !!! error opening output header file %s !!!\n", FileHdr);
        }

	for (i=0; i<MAXPICK; ++i) picks[i]=0.0;

	/* loop over traces */
	nrec = 0;
	recno = -999;
	while (gettr(&tr)) {
		if (!fh.ntraces) { /* starting at first trace */
			fh.npts=tr.ns;
			fh.sint=tr.dt;

	        	if (!getparint("npick", &fh.npick))	fh.npick  = MAXPICK;
	        	if (!getparint("tstart", &fh.tstart))	fh.tstart = tr.stas;
			if (!getparint("tend", &fh.tend))	fh.tend   = tr.stae;
	        	if (!getparfloat("vred", &fh.vredf))    fh.vredf  = tr.swevel/1000.0;

			nbytes=(22 + fh.npick)*4;

			/* write file header */
			fwrite(&fh, 4, 22, fpout);
			fwrite(picks, sizeof(float), fh.npick, fpout);
			fwrite(tr.data, sizeof(float), fh.npts, fpout);

			if(verbose) {
		    	printf("\n Number of samples = %d, Sampling int.= %5.3f (s), \
				\n Trace start/end = %5.3f/%5.3f (s), Vred = %5.3f (km/s)\n",
				fh.npts, fh.sint/1000000., fh.tstart/1000., fh.tend/1000.,
				fh.vredf);
			}
		}

		++fh.ntraces;

		gethval(&tr,index2, &val2);
 		gethval(&tr,index3, &val3);
 		gethval(&tr,index4, &val4);		
		ival2 = vtoi(type2, val2);
		ival3 = vtoi(type3, val3);
		ival4 = vtoi(type4, val4);
	    switch(ival4) {
	       case 12: th.itype=1; break;
	       case 13: th.itype=3; break;
	       case 14: th.itype=2; break;
	       case 15: th.itype=1; break;
	       case 16: th.itype=3; break;
	       case 17: th.itype=2; break;
	       case 11: th.itype=4; break;
	       default: printf("Default data type is hydrophone\n"); break;
	    }		
		if ( recno != ival2 ) /* change of rec */
		{
			if (nrec)
			{
				if(verbose) printf("\n\tTotal %d traces for profile %d record %d\n",
							th.itsn, ival1, th.nrec);
			}
			++nrec;
			recno = ival2;
			th.nrec=recno;
			th.itsn=0;  /* reset trace no within rec */

			gethval(&tr,index1, &val1);
			ival1 = vtoi(type1, val1);

			if(verbose > 1) {
			    printf("\n\tProfNo RecNo ShotNo TraceNo Offset(km) Azimuth\n");
			}
		}
		++th.itsn;
		/*th.itsn = ival3*/;
		th.ireci=ival3;
		th.offset=tr.offset;
		th.azi=tr.cdpt/60.0;
		th.slat=( abs(tr.scalco) < 10 ) ?
			tr.sy*pow(10,tr.scalco)/3600.0 : tr.sy/abs(tr.scalco);
		th.slong=( abs(tr.scalco) < 10 ) ?
			tr.sx*pow(10,tr.scalco)/3600.0 : tr.sx/abs(tr.scalco);
		th.selev=tr.selev;
		th.swdepth=tr.swdep;
		th.rlat=( abs(tr.scalco) < 10 ) ?
			tr.gy*pow(10,tr.scalco)/3600.0 : tr.gy/abs(tr.scalco);
		th.rlong=( abs(tr.scalco) < 10 ) ?
			tr.gx*pow(10,tr.scalco)/3600.0 : tr.gx/abs(tr.scalco);
		th.relev=tr.gelev;
		th.sxutm=tr.sx;
		th.syutm=tr.sy;
		th.rxutm=tr.gx;
		th.ryutm=tr.gy;
/*        printf("%d %d %d\n",th.relev,th.relev,th.selev);
		/* write trace header + trace data */

		fwrite(&nbytes, 4, 1, fphdr);
		fwrite(&th, 4, 22, fphdr);
		fwrite(picks, sizeof(float), fh.npick, fphdr);
		fwrite(&nbytes, 4, 1, fphdr);

		fwrite(&th, 4, 22, fpout);
		fwrite(picks, sizeof(float), fh.npick, fpout);
		fwrite(tr.data, sizeof(float), fh.npts, fpout);

		if(verbose > 1) {
		    printf("\t%6d%6d%7d%8d%11.3f%8.3f\n",
			ival1, th.nrec, th.ireci, th.itsn, th.offset/1000., th.azi/60.);
		}
	}

	/* write number of traces at the beginning of file */

	fseek(fpout, 0, SEEK_SET);
	fwrite(&fh.ntraces, 4, 1, fpout);
	if (fh.nrec == 0)
	{
		fseek(fpout, 5*sizeof(int), SEEK_SET);
		fwrite(&nrec, 4, 1, fpout);
	}

	if(verbose) printf("\n\tTotal %d traces for profile %d record %d\n", th.itsn, ival1, th.nrec);
	if(verbose) printf("\n %d records with totally %d traces \n", nrec, fh.ntraces);

	fclose(fpout);
	fclose(fphdr);

	return EXIT_SUCCESS;
}


