c
c
c     A program to update the active header file with the new picks and
c     dead trace flags
c
c     Written by C.A. Zelt  -  February 1994  -  Bullard Labs, Cambridge, UK.
c
      include 'zplot.par'
      character*80 hfile
      real picks(ntmax,npickw),offset(ntmax),az(ntmax),texact(ntmax),
     +     slat(ntmax),slong(ntmax),rlat(ntmax),rlong(ntmax)
      integer ishot(ntmax),itsn(ntmax),irec(ntmax),itype(ntmax),
     +     iflag(ntmax),apick,gain(ntmax),selev(ntmax),swdepth(ntmax),
     +     relev(ntmax),sxutm(ntmax),syutm(ntmax),sz(ntmax),
     +     rxutm(ntmax),ryutm(ntmax),rz(ntmax)

c
      write(*,*) 'Enter header file name:/'
      read(5,1) hfile
1     format(a80)
c
      open(16, file=hfile, form='unformatted', status='old')
c
      write(*,*) 'Enter number of pick words'
      read(*,*) npick
c
      i=1
100   read(16,end=999) ishot(i),itsn(i),irec(i),itype(i),iflag(i),
     +           offset(i),az(i),gain(i),texact(i),slat(i),slong(i),
     +           selev(i),swdepth(i),rlat(i),rlong(i),relev(i),
     +           sxutm(i),syutm(i),sz(i),rxutm(i),ryutm(i),rz(i),
     +           (picks(i,j),j=1,npick)
      i=i+1
      go to 100
999   ntrace=i-1
c
      write(*,*) 'number of traces: ',ntrace
c
      open(17, file='zplot.out', status='old')
c
      nup=0
101   read(17,15,end=998) ishotu,itsnu,apick,picku
15    format(3i6,f12.3)
c
      nup=nup+1
      ifound=0
      do 10 i=1,ntrace
c
         if(ishotu.eq.ishot(i).and.itsnu.eq.itsn(i)) then
           if(apick.eq.0) then
             iflag(i)=nint(picku)
           else
             if(apick.gt.0.and.apick.le.npick) then
               picks(i,apick)=picku
             else
               write(*,*) '*** invalid pick word number  ***'
               stop
             end if
           end if
           ifound=ifound+1
         end if
c
10    continue
c
      if(ifound.eq.0) then
        write(*,*) '**  trace not in header file  ***'
        stop
      end if
c
      if(ifound.gt.1) then
        write(*,*) '***  more than one trace with same shot & tsn  ***'
        stop
      end if
c
      go to 101
c
998   write(*,*) 'total updates is:',nup
c
      rewind(16)
c
      do 20 i=1,ntrace
c
         write(16) ishot(i),itsn(i),irec(i),itype(i),iflag(i),
     +    offset(i),az(i),gain(i),texact(i),slat(i),slong(i),
     +           selev(i),swdepth(i),rlat(i),rlong(i),relev(i),
     +           sxutm(i),syutm(i),sz(i),rxutm(i),ryutm(i),rz(i),
     +           (picks(i,j),j=1,npick)
c
20    continue
c
      stop
      end
