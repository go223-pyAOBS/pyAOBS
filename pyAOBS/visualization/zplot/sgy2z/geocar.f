C
      SUBROUTINE GEOCAR(LAT,LONG,X,Y)
C
C
C  This subroutine calculates cartesian coordinates from latitude
C and longitude using a transverse mercator projection. The origin
C of the cartesian coordinate system is at the intersection of the
C projections central meridian and the equator.  X increases to the
C east from the central meridian, and Y increases to the north from
C the equator.
C Note that all calculations are done using double precision.
C
C The calling program must contain the following lines:
C 
C      REAL*8 L0, R, ESQ, K0
C      COMMON /MERC/L0,R,ESQ,K0
C 
C      L0=129.0D0
C      R=6.3782064D6
C      ESQ=6.768657997D-3
C      K0=0.9996D0 (normally; sometimes 1.0D0)

C
C   LAT.......latitude of the position ( decimal degrees ).
C
C   LONG......longitude of the position ( decimal degrees ).
C
C   L0........central meridian (in decimal degrees).
C
C   R.........radius of the earth in metres.
C
C   ESQ.......square of the earth's first eccentricity.
C
C   X.........calculated x-coordinate of projected point ( metres ).
C
C   Y.........calculated y-coordinate of projected point ( metres ).
C
C
C
      IMPLICIT REAL*8(A-Z) 
      COMMON /MERC/L0,R,ESQ,K0
      EPRMSQ=ESQ/(1.0D0-ESQ) 
      SINSEC=3.1415926535D0/180.0D0 
C 
C     Convert latitude to radian measure so that FORTRAN trigonometric 
C     functions can be used. 
C
      CL=LAT*SINSEC 
 
C
C     NOTE THE CONVENTION  -180 < LONG < 180 
C 
c      DLONG=(L0-LONG)
      DLONG=(LONG-L0)
      NU=R/DSQRT(1.0D0-ESQ*DSIN(CL)**2) 
      SA=(1.0D0+(3.0D0/4.0D0)*ESQ+(45.0D0/64.0D0)*ESQ**2+ 
     &(175.0D0/256.0D0)*ESQ**3)*CL 
      SB=(((3.0D0/4.0D0)*ESQ+(15.0D0/16.0D0)*ESQ**2+(525.0D0/512.0D0)* 
     &ESQ**3)*DSIN(2.0D0*CL))/2.0D0 
      SC=(((15.0D0/64.0D0)*ESQ**2+(105.0D0/256.0D0)*ESQ**3)* 
     &DSIN(4.0D0*CL))/4.0D0 
      SD=((35.0D0/512.0D0)*ESQ**3*DSIN(6.0D0*CL))/6.0D0 
      S=R*(1.0D0-ESQ)*(SA-SB+SC-SD) 
      I=S*K0 
      II=(NU*DSIN(CL)*DCOS(CL)*SINSEC**2*K0)/2.0D0 
      IIIA=(SINSEC**4*NU*DSIN(CL)*DCOS(CL)**3)/24.0D0 
      IIIB=(5.0D0-DTAN(CL)**2+9.0D0*EPRMSQ*DCOS(CL)**2+4.0D0*EPRMSQ**2 
     &*DCOS(CL)**4)*K0 
      III=IIIA*IIIB 
      IV=NU*DCOS(CL)*SINSEC*K0 
      V=(SINSEC**3*NU*DCOS(CL)**3*(1.0D0-DTAN(CL)**2+EPRMSQ*DCOS(CL) 
     &**2)*K0)/6.0D0 
      A6A=(DLONG**6*SINSEC**6*NU*DSIN(CL)*DCOS(CL)**5)/7.2D2 
      A6B=(61.0D0-58.0D0*DTAN(CL)**2+DTAN(CL)**4+2.7D2*EPRMSQ*DCOS(CL 
     &)**2-3.3D2*EPRMSQ*DSIN(CL)**2)*K0 
      A6=A6A*A6B 
      B5A=(DLONG**5*SINSEC**5*NU*DCOS(CL)**5)/1.2D2 
      B5B=(5.0D0-18.0D0*DTAN(CL)**2+DTAN(CL)**4+14.0D0*EPRMSQ*DCOS(CL 
     &)**2-58.0D0*EPRMSQ*DSIN(CL)**2)*K0 
      B5=B5A*B5B 
      Y=I+II*DLONG**2+III*DLONG**4+A6 
      X=IV*DLONG+V*DLONG**3+B5 
      RETURN 
      END 
