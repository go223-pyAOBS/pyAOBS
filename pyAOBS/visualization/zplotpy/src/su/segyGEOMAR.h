/* This is the definition of the segy trace header for the
   GEOMAR landstation and OBS wide-angle reflection data. The extension of the
   standard SEGY header from 181 to 240 byte is layout in order to process
   the data on the GEOSYS software system.  
   
   Reading bytes directly into this header will allow access 
   to all of the fields.

   New version 95 is thought to be better integrated in the
   processing package Seismic Unix

*/
struct SegyHeadGEOMAR {
  long lineSeq, reelSeq; /* Sequence numbers within line and reel, resp. */
                         /* here station and shot number Def: 1, 1 */
  long profNumber;       /* Original field record number */
                         /* Here profile number */
  long traceNumber;      /* Trace number within the original field record.*/
                         /* Here station (receiver) Number  */
  long energySourcePt;   /* Energy source (shot) point number          20 */
                         /* Def: 0 */
  long cdpEns;           /* CDP ensemble number: shot number */
                         /* Def: 0 */
  long traceInEnsemble;  /* Trace number within CDP ensemble */
                         /* Here azimuth in second of arc  */
  short traceID;         /* Trace identification code:
                  1=seismic data (Def)  4=time break   7=timing
                  2=dead                5=uphole       8=water break
                  3=dummy               6=sweep        9..., optional use  */
  short vertSum, horSum;                 /* Def: 1, 1 */
  short dataUse;                         /* 1=production (Def), 2=test */
  long sourceToRecDist;                  /* Distance in (m)               40 */
  long recElevation;                     /* Elevation in (m), Def: 0  */ 
  long sourceSurfaceElevation;           /* Def: 0 (m)  */
  long sourceDepth;                      /* Def: 0 (m)  */
  long datumElevRec, datumElemSource;    /* Def: 0, 0 (m)                 60 */
  long sourceWaterDepth, recWaterDepth;  /* Def: 0, 0 (m) */
  short elevationScale;                  /* Scale elevations Def: 0 (10**0) */
  short coordScale;                      /* Scale coordnates
                                            Def: -2, means coordinates multiplied
                                            by 10**(-2) to get real value*/
  long sourceLongOrX, sourceLatOrY;      /* Either cartesian or geographic80 */
  long recLongOrX, recLatOrY; 
  short coordUnits;                      /* 1= meter or feet; 2=sec of arc */
  short weatheringVelocity;              /* Def: 0 (m/s) */ 
  short subWeatheringVelocity;           /* Reduction velocity, Def: 6000 (m/s) */
  short sourceUpholeTime;  		 /* Def: 0 (ms) */
  short recUpholeTime; 			 /* Def: 0 (ms) */
  short sourceStaticCor, recStaticCor;   /* Def: 0, 0  (ms)              102 */
  short totalStatic;                     /* Def: 0 (ms) */
  short lagTimeA;                        /* T(shottime) - T(first sample)  */
  short lagTimeB;                        /* Def: 0 (ms) */ 
  short delay;                           /* Def: 0 (ms) */
  short muteStart, muteEnd;              /* Def: 0, 0 (ms) */
  short sampleLength;            /* Number of samples in this trace */
                                 /* ( > 32767 )? = 32767  
                                    set long samp_rate in 185-188 byte */
  short deltaSample;             /* Sampling interval in microseconds. */
  short gainType;                /* 1=fixed (Def), 2=binary,             120
                                    3=floating, 4... opt.*/
  short gainConst;               /* Gain of recording channel  */
  short initialGain;             /* Gain of preamplifier in db */
  short correlated;              /* 1=no (Def), 2=yes */
  short sweepStart, sweepEnd;    /* min. and max. amplitude of trace */
  short sweepLength;             /* Here defined as 
                                    fraction of second of shot time  */
  short sweepType;               /* Source type:
                     1=linear,  2=parabolic,  3=exponential,  4=others
                     5=bohrhole explosive,  6=water explosive,  7=airgun (Def)*/
  short sweepTaperAtStart, 
        sweepTaperAtEnd;         /* Start and end of trace (ms) relative to Tred(0) */
  short taperType;               /* scaling factor for last two values Def: 1 (x10) 140 */
  short aliasFreq, aliasSlope;   /* Def: 0, 0  */
  short notchFreq, notchSlope;   /* Def: 0, 0  */
  short lowCutFreq, hiCutFreq;   /* Def: 0, 0  */
  short lowCutSlope, hiCutSlope; /* Def: 0, 0  */                    /* 156 */
  short year, day, hour,         /* Source (shot) time, the fraction of sec */
        minute, second;          /* is set in millisec between 131-132 byte */
  short timeBasisCode;           /* 1=local, 2=GMT, 3=MET (GMT + 1 hour) (Def) */
  short traceWeightingFactor;    /*  */
  short phoneRollPos1; 		 /* Component: 1=time code, 2=radial, 3=transverse
                                    4=vertical, 5=hydrophone (Def)  */
  short phoneFirstTrace; 	 /* Methusalem instrument number in YYNN */
  short phoneLastTrace;		 /* Channel number */
  short gapSize;		 /* Source charge 
				    in cubic inches (airgun) or kg (explosives) 170 */
  short taperOvertravel;         /* Def: 0=meaningless  1=up, 2=down   180 */ 
                                 /* !!! Following is extension !!! */
  short compNo;                  /* 1=time code, 2=radial, 3=transverse
                                    4=vertical, 5=hydrophone (Def)  */
  short samplingRate;            /* samples/sec */
  long  numberSamples;           /* ( <= 32767) ? sampleLength | ( > 32767) */
  short shotPointNo;                                                 /* 190 */
  short ADCoeff;                 /* Coefficient of A/D converter in mv/digit */
  short receiverCoeff;           /* Coversion coefficient of receiver, 
                                    pascal/cm2 for hydrophone, 
                                    velocity(m/s)/volt for geophone */
  short receiverType;            /* 1=hydrophone (Def), 2=geophone, 3...    */
  long lengthData;               /* Def: 0 (ms), not used here          200 */
  long  distance;                /* Source to receiver distance in (m)  */
  float scaleFactor;             /* Scale factor same as in <segy.h> */
  short azimuth;                 /* Orientation of the component in min 210 */
  short eigenperiod;             /* Eigenperiod of geo- or hydrophone in (ms) */
  long  minAmpl;                 /* Min. peak amplitude within trace     */
  long  maxAmpl;                 /* Max. peak amplitude within trace    220 */
  short stationNo;               /* Station number */
  short channelNo;               /* Channel number (Default: 1) */
  long  sourceCharge;            /* Charge in kg (explosive) or cc (airgun) */
  short redVelocity;             /* reduction velocity in (m/s); 
                                    Def: 0 if no reduction velocity set 230 */
  short timeOffset;              /* Time offset in (ms) of first sample
                                    relative to reduced source time: 
                                    positive if earlier than reduced time  */
  long  redTime;                 /* Reduced time in (ms) = distance/redVel  */
  short unused2;
  short instNo;                  /* Methusalem instrument number */
};                               /* end of segy trace header */
