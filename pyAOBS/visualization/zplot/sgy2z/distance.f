c
c     version 1.0  Feb 1994
c
c     distance/azimuth routines for SGY2Z
c
c     ----------------------------------------------------------------
c
      subroutine distaz(rla1,rlo1,sla1,slo1,dd,zz)
c
      implicit none
c
c     this version assumes east positive (geodist doesn't)
c
      real rla1,rlo1,sla1,slo1,dd,zz
      double precision rla,rlo,sla,slo
      double precision slat,slon,elat,elon,d,az1,az2,degr
c
      rla=rla1
      rlo=rlo1
      sla=sla1
      slo=slo1
c
      degr=datan(1.d0)/45.d0
c
      if(rlo.gt.180.e0)rlo=rlo-360.e0
      if(slo.gt.180.e0)slo=slo-360.e0
      if(rlo.lt.-180.e0)rlo=rlo+360.e0
      if(slo.lt.-180.e0)slo=slo+360.e0
c
      elat=sla*degr
      elon=-slo*degr
      slat=rla*degr
      slon=-rlo*degr
c
      call geodis(slat,slon,elat,elon,d,az1,az2)
c
      zz=az1/degr
      dd=d
c
      return
      end
c
c     ----------------------------------------------------------------
c
      subroutine geodis (slat,slon,elat,elon,d,az1,az2)              
      implicit double precision (a-h,o-z)
c     uses geodetic inverse formula (geodesy by a r clarke)          
c     slat,slon are station latitude and longitude in radians        
c     elat,elon are shot latitude and longitude in radians           
c     d is the distance in kilometres                                
c     az1 is the azimuth of the station from the shot in radians     
c     az2 is the azimuth of the shot from the station in radians     
      dimension c(4)                                                 
      able = 6378.2064d0                                             
      bake = 6356.5838d0                                             
      easy = 0.006768658d0                                           
      rat = bake*bake/(able*able)                                    
      cslat = cos (slat)                                             
      sslat = sin (slat)                                             
      celat = cos (elat)                                             
      selat = sin (elat)                                             
      cd = cos (slon - elon)                                         
      sd = sin (slon - elon)                                         
      ens = able/sqrt (1.d0-easy*sslat*sslat)                        
      ene = able/sqrt (1.d0-easy*selat*selat)                        
      ene = ene/ens                                                  
      a = -cslat*sd                                                  
      b = rat*celat*sslat-cslat*selat*cd+easy*celat*selat*ene        
      az1 = atang(a,b)                                               
      a = celat*sd                                                   
      b = rat*cslat*selat-celat*sslat*cd+easy*cslat*sslat/ene        
      az2 = atang(a,b)                                               
      fkon = celat*cd*ene-cslat                                      
      a = celat*sd*ene                                               
      b = selat*ene-sslat                                            
      b = b*rat                                                      
      fkon = fkon*fkon+a*a+b*b                                       
      fkon = sqrt (fkon)                                             
      b = sqrt (easy/(1.-easy))                                      
      a = b*sslat                                                    
      b = b*cslat*cos (az2)                                          
      beff = 1.d0+b*b                                                
      fkor = fkon*beff                                               
      beff = 1.d0/beff                                               
      bach = (a*a-b*b)*beff                                          
      beff = a*b*beff                                                
      c(1) = -0.1875d0*beff*bach-0.3333333333d0*beff*beff*beff       
      c(2) = 0.0046875d0+.0375d0*bach+0.25d0*beff*beff               
      c(3) = -0.125d0*beff                                           
      c(4) = 0.04166666667d0                                         
      d = c(1)                                                       
      do 1 k = 2,4                                                   
      d = d*fkor+c(k)                                                
    1 continue                                                       
      d = d*fkor*fkor+1.                                             
      d = fkon*ens*d                                                 
      return                                                         
      end                                                            
c
c     ----------------------------------------------------------------
c
      function atang(a,b)   
      implicit double precision (a-h,o-z) 
c     computes atan(a/b) for all values of a and b, from 0 to 2*pye  
      if(b)1,3,5                                                     
    1 atang = atan (a/b) + 3.14159265d0                              
      return                                                         
    3 if(a)7,9,11                                                    
    7 atang=4.71238897d0                                             
      return                                                         
    9 atang = 0.                                                     
      return                                                         
   11 atang = +1.57079632d0                                          
      return                                                         
    5 if(a)13,15,15                                                  
   13 atang = atan (a/b) + 6.2831853d0                               
      return                                                         
   15 atang = atan (a/b)                                             
      return                                                         
      end                                                            
