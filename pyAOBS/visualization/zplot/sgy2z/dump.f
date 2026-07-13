c
      subroutine dump_hd1(header,idum2,idum4,cfact)
c
      integer*2 idum2(194)
      integer*4 idum4(3)
      character*80 header(40)
c
      write(8,'(''________ BINARY REEL HEADER ________'')')
      write(8,'(''Line number                         '',I8)')idum4(2)
      write(8,'(''No of traces per record             '',I8)')idum2(1)
      write(8,'(''Sample interval (millisecs)         '',f8.4)')
     +  idum2(3)/1000.
      write(8,'(''No of samples per trace             '',I8)')idum2(5)
      write(8,'(''Data sample format code             '',I8)')idum2(7)
      write(8,'(''Trace sorting code                  '',I8)')idum2(9)
c
      return
      end
c
      subroutine dump_trhd1(tr_long1,tr_short1, tr_long2,tr_short2,
     +                      tr_long3,tr_short3,cfact)
c
      integer*4 tr_long1(7)
      integer*4 tr_long2(8)
      integer*4 tr_long3(4)
      integer*2 tr_short1(4)
      integer*2 tr_short2(2)
      integer*2 tr_short3(76)
c
      write(8,'(''___________ TRACE HEADER ___________'')')
      write(8,'(''Trace sequence number within line '',I10)')tr_long1(1)       
      write(8,'(''Trace sequence number within reel '',I10)')tr_long1(2)
      write(8,'(''Trace id code                     '',I10)')
     +  tr_short1(1)
      write(8,'(''Data use                          '',I10)')
     +  tr_short1(4) 
      write(8,'(''Source to receiver offset (km)    '',f10.3)')
     +  tr_long2(1)/1000.
      write(8,'(''Surface elevation at source (km)    '',f8.3)')
     +  tr_long2(3)/1000.
      write(8,'(''Water depth at source (km)      '',f12.3)')
     +  tr_long2(7)/1000.
      write(8,'(''Water depth at OBS (km)         '',f12.3)')
     +  tr_long2(8)/1000.
      write(8,'(''Source longitude (degrees)        '',f10.5)')
     +  tr_long3(1)/cfact
      write(8,'(''Source latitude  (degrees)        '',f10.5)')
     +  tr_long3(2)/cfact
      write(8,'(''OBS longitude (degrees)           '',f10.5)')
     +  tr_long3(3)/cfact
      write(8,'(''OBS latitude  (degrees)           '',f10.5)')
     +  tr_long3(4)/cfact
      write(8,'(''Delay recording time (millisecs) '',I11)')
     +  tr_short3(11)
      write(8,'(''Number of samples in this trace  '',I11)')
     +  tr_short3(14) 
      write(8,'(''Sample interval (millisecs)      '',f11.4)')
     +  tr_short3(15)/1000.
      write(8,'(''Instrument gain constant         '',I11)')
     +  tr_short3(17) 
c
      return
      end
c
      subroutine dump_hd2(header, idum2, idum4,cfact)
      integer*2 idum2(194)
      integer*4 idum4(3)
      character*80 header(40)

      write(8,'(''___ CHARACTER REEL HEADER ___'')')
      write(8,*) (header(j),j=1,40)
      write(8,'(''________ BINARY REEL HEADER ________'')')
      write(8,'(''Job id                              '',I8)')idum4(1)
      write(8,'(''Line number                         '',I8)')idum4(2)
      write(8,'(''Reel number                         '',I8)')idum4(3)
      write(8,'(''No of traces per record             '',I8)')idum2(1)
      write(8,'(''No of auxiliary traces per record   '',I8)')idum2(2)
      write(8,'(''Sample interval (millisecs)         '',f8.4)')
     +  idum2(3)/1000.
      write(8,'(''Field sample interval (microsecs)   '',I8)')idum2(4)
      write(8,'(''No of samples per trace             '',I8)')idum2(5)
      write(8,'(''Field no of samples per trace       '',I8)')idum2(6)
      write(8,'(''Data sample format code             '',I8)')idum2(7)
      write(8,'(''CDP fold                            '',I8)')idum2(8)
      write(8,'(''Trace sorting code                  '',I8)')idum2(9)
      write(8,'(''Vertical sum code                   '',I8)')idum2(10)
      write(8,'(''Sweep frequency - start             '',I8)')idum2(11)
      write(8,'(''Sweep frequency - end               '',I8)')idum2(12)
      write(8,'(''Sweep length (ms)                   '',I8)')idum2(13)
      write(8,'(''Sweep type code                     '',I8)')idum2(14)
      write(8,'(''Trace number of sweep channel       '',I8)')idum2(15)
      write(8,'(''Sweep trace taper length (ms )      '',I8)')idum2(16)
      write(8,'(''Taper type                          '',I8)')idum2(17)
      write(8,'(''Correlated data traces              '',I8)')idum2(18)
      write(8,'(''Binary gain recovered               '',I8)')idum2(19)
      write(8,'(''Amplitude recovery method           '',I8)')idum2(20)
      write(8,'(''Measurement system                  '',I8)')idum2(21)
      write(8,'(''Impulse signal polarity             '',I8)')idum2(22)
      write(8,'(''Vibratory polarity code             '',I8)')idum2(23)
      return
      end
c
      subroutine dump_trhd2(tr_long1, tr_short1, 
     +                    tr_long2,  tr_short2,
     +                    tr_long3,  tr_short3,cfact)

      integer*4 tr_long1(7)
      integer*4 tr_long2(8)
      integer*4 tr_long3(4)
      integer*2 tr_short1(4)
      integer*2 tr_short2(2)
      integer*2 tr_short3(76)
      write(8,'(''___________ TRACE HEADER ___________'')')
      write(8,'(''Trace sequence number within line '',I10)')tr_long1(1)       
      write(8,'(''Trace sequence number within reel '',I10)')tr_long1(2)
      write(8,'(''Original field record no.         '',I10)')tr_long1(3)
      write(8,'(''Trace no. within field record     '',I10)')tr_long1(4)
      write(8,'(''Energy source point no.           '',I10)')tr_long1(5) 
      write(8,'(''CDP ensemble no.                  '',I10)')tr_long1(6)
      write(8,'(''Trace no. within CDP ensemble     '',I10)')tr_long1(7) 
      write(8,'(''Trace id code                     '',I10)')
     +  tr_short1(1)
      write(8,'(''No. of vertically summed traces   '',I10)')
     +  tr_short1(2)
      write(8,'(''No. of horizontally stacked traces'',I10)')
     +  tr_short1(3) 
      write(8,'(''Data use                          '',I10)')
     +  tr_short1(4) 
      write(8,'(''Source to receiver offset (km)    '',f10.3)')
     +  tr_long2(1)/1000.
      write(8,'(''Receiver group elevation          '',I9)')tr_long2(2)
      write(8,'(''Surface elevation at source (km)    '',f8.3)')
     +  tr_long2(3)/1000.
      write(8,'(''Source depth                      '',I8)')tr_long2(4)
      write(8,'(''Datum elevation at receiver       '',I8)')tr_long2(5) 
      write(8,'(''Datum elevation at source         '',I8)')tr_long2(6)
      write(8,'(''Water depth at source (km)      '',f12.3)')
     +  tr_long2(7)/1000.
      write(8,'(''Water depth at OBS (km)         '',f12.3)')
     +  tr_long2(8)/1000.
      write(8,'(''Scaler for elevations & depths    '',I8)')tr_short2(1) 
      write(8,'(''Scaler for co-ords                '',I8)')tr_short2(2)
      write(8,'(''Source longitude (degrees)        '',f10.5)')
     +  tr_long3(1)/cfact
      write(8,'(''Source latitude  (degrees)        '',f10.5)')
     +  tr_long3(2)/cfact
      write(8,'(''OBS longitude (degrees)           '',f10.5)')
     +  tr_long3(3)/cfact
      write(8,'(''OBS latitude  (degrees)           '',f10.5)')
     +  tr_long3(4)/cfact
      write(8,'(''Co-ordinate units                 '',I8)')tr_short3(1) 
      write(8,'(''Weathering velocity               '',I8)')tr_short3(2)
      write(8,'(''Subweathering velocity            '',I8)')tr_short3(3) 
      write(8,'(''Uphole time at source             '',I8)')tr_short3(4) 
      write(8,'(''Uphole time at group              '',I8)')tr_short3(5) 
      write(8,'(''Source static correction        '',I8)')tr_short3(6) 
      write(8,'(''Group static correction           '',I8)')tr_short3(7)
      write(8,'(''Total static applied              '',I8)')tr_short3(8) 
      write(8,'(''Lag time A                        '',I8)')tr_short3(9)
      write(8,'(''Lag time B                       '',I8)')tr_short3(10) 
      write(8,'(''Delay recording time (millisecs) '',I11)')
     +  tr_short3(11)
      write(8,'(''Mute time - start                '',I8)')tr_short3(12) 
      write(8,'(''Mute time - end                  '',I8)')tr_short3(13)
      write(8,'(''Number of samples in this trace  '',I11)')
     +  tr_short3(14) 
      write(8,'(''Sample interval (millisecs)      '',f11.4)')
     +  tr_short3(15)/1000.
      write(8,'(''Gain type of field instruments   '',I8)')tr_short3(16) 
      write(8,'(''Instrument gain constant         '',I11)')
     +  tr_short3(17) 
      write(8,'(''Instrument initial gain (dB)     '',I8)')tr_short3(18)
      write(8,'(''Correlated ( 1 = yes, 2 = no )   '',I8)')tr_short3(19) 
      write(8,'(''Sweep frequency at start         '',I8)')tr_short3(20)
      write(8,'(''Sweep frequency at end           '',I8)')tr_short3(21) 
      write(8,'(''Sweep length in ms               '',I8)')tr_short3(22) 
      write(8,'(''Sweep type                       '',I8)')tr_short3(23)
      write(8,'(''Sweep trace taper len at start ms'',I8)')tr_short3(24)     
      write(8,'(''Sweep trace taper len at end ms  '',I8)')tr_short3(25) 
      write(8,'(''Taper type                       '',I8)')tr_short3(26)
      write(8,'(''Alias filter frequency           '',I8)')tr_short3(27) 
      write(8,'(''Alias filter slope               '',I8)')tr_short3(28) 
      write(8,'(''Notch filter frequency           '',I8)')tr_short3(29)
      write(8,'(''Notch filter slope               '',I8)')tr_short3(30) 
      write(8,'(''Low cut frequency                '',I8)')tr_short3(31) 
      write(8,'(''High cut frequency               '',I8)')tr_short3(32)  
      write(8,'(''Low cut slope                    '',I8)')tr_short3(33)
      write(8,'(''High cut slope                   '',I8)')tr_short3(34) 
      write(8,'(''Year data recorded               '',I8)')tr_short3(35) 
      write(8,'(''Day of year                      '',I8)')tr_short3(36)  
      write(8,'(''Hour of day                      '',I8)')tr_short3(37) 
      write(8,'(''Minute of hour                   '',I8)')tr_short3(38) 
      write(8,'(''Second of minute                 '',I8)')tr_short3(39)  
      write(8,'(''Time basis code                  '',I8)')tr_short3(40)
      write(8,'(''Trace weighting factor           '',I8)')tr_short3(41)
      write(8,'(''Geophone grp. no. roll switch 1  '',I8)')tr_short3(42)
      write(8,'(''Geophone grp. no. trace 1 (field)'',I8)')tr_short3(43)
      write(8,'(''Geophone grp. no. last trace     '',I8)')tr_short3(44)
      write(8,'(''Gap size                         '',I8)')tr_short3(45)
      write(8,'(''Overtravel                       '',I8)')tr_short3(46)
      write(8,'(''Optional data (assumed to be short integers:'')') 
      do 15, i = 47, 76
        write(8,'(''cols ''i3,'' to '', i3,'' - '', i2)')
     +                88 + i*2-1,       88+i*2,        tr_short3(i)
  15  continue 
c
      return
      end
