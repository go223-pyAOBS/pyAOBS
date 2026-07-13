
       if(kclass.eq.'u') then
		 nupt=2
         go to 119
       end if
	   
      if(kclass.eq.'g'.or.kclass.eq.'t') then
	    if(kclpre.ne.'u'.or.nupt.eq.3) nupt=1
		if(nupt.eq.2) then
		 irec_now=irec
         if(kclass.eq.'t'.and.iplt.gt.2)then
		 ialigu=1
		 title='Set time baseline'
		 
         if(itrev.ne.1) then
           taligu=(y-orig)*tscale+tminm
         else
           taligu=(y-orig)*tscale+tmaxm
         end if
         do 707 ita=1,iplt
c
            itn=iposp(ita)
            igun=ireci(itn)
			if(iflagi(itn).gt.0) then
             if(picks(itn,apick).gt.-998..and.
     +         picks(itn,apick).ne.0.) then
			  tstatic(igun)=picks(itn,apick)-pcorr(itn)+abs(offsti(itn))*
     +              (-rvredm)-taligu
	         end if
			end if
707      continue
		 end if
		 if(kclass.eq.'u')then
		 ialigu=0
		 title='Release time baseline'
		 end if
		 
		 goto 1113
		end if
		nupt=nupt+1
        go to 119
      end if	  