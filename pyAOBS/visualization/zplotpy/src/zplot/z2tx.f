c Copyright (c) GEOMAR, 1997. 
c All rights reserved.                      

c modify by Haibo  20240522 
c   -------   for select different OBS 
c           tshift(nsmax) for each OBS 
c            nshot(nsmax) for each OBS 
c       amscal(nsmax) for diffrent OBS 
c
      include 'zplot.par'
      character*80 hfile,ofile
      real picks(ntmax,npickw),offset(ntmax),az(ntmax),texact(ntmax),
     +     slat(ntmax),slong(ntmax),rlat(ntmax),rlong(ntmax)
      integer ishot(ntmax),itsn(ntmax),irec(ntmax),itype(ntmax),
     +     iflag(ntmax),gain(ntmax),selev(ntmax),swdepth(ntmax),
     +     relev(ntmax),sxutm(ntmax),syutm(ntmax),sz(ntmax),
     +     rxutm(ntmax),ryutm(ntmax),rz(ntmax),iamp,nshot(nsmax),
     +     nsec,nsec1,isall
c
      real xmod(nsmax),picku(npickw),azimuth(nsmax),xshift(nsmax)
     +    ,tshift(nsmax),amscal(nsmax),pshot(nsmax),pick
c
      namelist /par/ hfile,ofile,npick,xmod,picku,azimuth,nshot,xshift,
     +              tshift,amscal,iamp
c
c
c     initialize parameters
c
      data picku/npickw*-1/,tshift/nsmax*0/,amscal/nsmax*1/
     +     xshift/nsmax*0/,nshot/nsmax*-1/
      open(10, file='z2tx.in', status='old')
      iamp=0
	  ofile='tx.in'
c
      read(10,par)
c
c     haibo
c     assign default value to picku if not specified or
c     picku(1) if only it is specified
c                 
      if(picku(1).lt.0) then
         do 320 i=1,npickw
            picku(i)=0.05
320      continue 
      else        
        if(picku(2).lt.0) then
          do 330 i=2,npickw
             picku(i)=picku(1)
330       continue
        else      
          do 340 i=1,npickw
             if(picku(i).lt.0) picku(i)=0.05
340       continue
        end if    
      end if
	  
      write(6,*) '######### OBSs you input in z2tx.in #########'
      write(6,*) 'OBS#      xmod    tshift    xshift    amscal'	  
      isall=0
       nsec=0	  
        do 370 i=1,nsmax
		   if(nshot(i).ge.0)then
		    nsec=nsec+1
			write(*,118)nshot(i),xmod(i),tshift(i),xshift(i),
     +                  amscal(i)
		   end if
370     continue		   
118   format(i5,4f10.3)
      open(16, file=hfile, form='unformatted', status='old')
c
      write(6,*) '###### OBSs included in your HDR file ######'
      write(6,*) 'OBS#     depth  shot_dep'
      i=1
	  nc=0
	  nowshot=-999
100   read(16,end=999) ishot(i),itsn(i),irec(i),itype(i),iflag(i),
     +           offset(i),az(i),gain(i),texact(i),slat(i),slong(i),
     +           selev(i),swdepth(i),rlat(i),rlong(i),relev(i),
     +           sxutm(i),syutm(i),sz(i),rxutm(i),ryutm(i),rz(i),
     +           (picks(i,j),j=1,npick)
      if(ishot(i).ne.nowshot) then
		nc=nc+1
   		nowshot=ishot(i)
		pshot(nc)=nowshot
		write(6,7) ishot(i),relev(i),selev(i)
7       format(i5,2i10)
	  end if
      i=i+1
      go to 100
999   ntrace=i-1
c
      write(6,*) '############################################'
      write(6,*) '       Number of traces: ', ntrace
	  write(6,*) '############################################'
      if(iamp.ne.0) write(6,1115) nc
	  if(iamp.eq.0) write(6,1116) nc
1115  format(' ****** ',i5,'     OBSs    converted   *******'/
     +' OBS#      xmod    amscal    xshift')
1116  format(' ****** ',i5,'     OBSs    converted   *******'/
     +' OBS#      xmod    tshift    xshift')	 
      do 50 i=1,nc
	    do 60 k=1,nsec
    	  if(pshot(i).eq.nshot(k))then
		   if(iamp.eq.0) write(6,8) nshot(k),xmod(k),
     +           tshift(k),xshift(k)
		   if(iamp.ne.0) write(6,8) nshot(k),xmod(k),
     +           amscal(k),xshift(k)	 
8         format(i5,3f10.3)
          end if
60      continue
50    continue
      write(6,*) '############################################'
c
      do 10 i=1,ntrace
         az(i)=az(i)/60.
         offset(i)=offset(i)/1000.
c        daz=abs(azimuth(ishot(i))-az(i))
c        if(daz.gt.90..and.daz.lt.270.) offset(i)=-offset(i)
10    continue
c
      open(11, file=ofile)
c
      npicks=0
c
      do 1000 i=1,nsec
         n1=0
         do 2001 j=1,npick
            do 3001 k=1,ntrace
               if(ishot(k).eq.nshot(i)) then
                 if(picks(k,j).gt.0..and.offset(k).lt.0.) then
                   if(n1.eq.0) then
                     write(11,5) xmod(i),-1.,0.,0
                     n1=1
                   end if
5                  format(3f10.3,i10)
c   haibo
                   if(iamp.ne.0) then
				     pick=picks(k,j)*amscal(i)
				   else
				     pick=picks(k,j)+tshift(i)
				   end if
				   
                   write(11,5) xmod(i)+offset(k)+xshift(i),
     +                         pick,picku(j),j
                   npicks=npicks+1
                 end if
               end if
3001        continue
2001     continue
         n1=0
         do 2002 j=1,npick
            do 3002 k=1,ntrace
               if(ishot(k).eq.nshot(i)) then
                 if(picks(k,j).gt.0..and.offset(k).gt.0.) then
                   if(n1.eq.0) then
                     write(11,5) xmod(i),1.,0.,0
                     n1=1
                   end if
c   haibo
                   if(iamp.ne.0) then
				     pick=picks(k,j)*amscal(i)
				   else
				     pick=picks(k,j)+tshift(i)
				   end if
				   
                   write(11,5) xmod(i)+offset(k)+xshift(i),
     +                         pick,picku(j),j
                   npicks=npicks+1
                 end if
               end if
3002        continue
2002     continue
1000  continue
c
      if(iamp.ne.0) then
		write(6,*) '******  Amplitudes picks are: ',npicks
	  else
		write(6,*) '******  Traveltime picks are ',npicks
	  end if
c
      if(npicks.gt.0) write(11,5) 0.,0.,0.,-1
c
      stop
      end
