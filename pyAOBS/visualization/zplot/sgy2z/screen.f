      SUBROUTINE PINPUT
C ***************************************************************
C *                                                             *
C *  This is a simple routine for prompting input, without the  *
C *  trouble of formatting.  A prompt text is written. If the   *
c *  default flag is set then the current value of the          *
C *  variable being prompted for is printed.                     *
C *  If a carriage return is the user response, the variable    *
C *  is left at the default value (the value on entry to the    *
C *  subroutine.  If something is actually entered, the         *
C *  variable is given this new value.  By calling this routine *
C *  under different names, REAL, INTEGER, or CHARACTER         *
C *  variables can be prompted for and obtained.                *
C *                                                             *
C *        CALL RIN(PROMPT,RVALUE) will print the text in       *
C *             character variable PROMPT and read a real       *
C *             value RVALUE.                                   *
C *        CALL DIN(PROMPT,DVALUE) is for double precision      *
C *             real values.                                    *
C *        CALL IIN(PROMPT,IVALUE) does the same except that    *
C *             it returns an integer value IVALUE.             *
C *        CALL TIN(PROMPT,TVALUE) does the same except that    *
C *             it returns a character value TVALUE.            *
C *        CALL NIN(PROMPT) merely prompts without reading      *
C *             any value.                                      *
C *                                                             *
C *  The character variables PROMPT and TVALUE can be of any    *
C *  length up to 80 for PROMPT and 132 for TVALUE.             *
C *  Writing is done on FORTRAN unit 6, reading from FORTRAN    *
C *  unit 5.                                                    *
C *                                                             *
C ***************************************************************
      CHARACTER*(*) PROMPT,TVALUE
      logical default
      CHARACTER*15 CODE,CODE0
      CHARACTER*132 RESPONSE
	real*8 dvalue, dlow, dhi
      DATA CODE0/'               '/
	entry din(prompt,dvalue, default, dlow, dhi)
      MODE=0
      CODE=CODE0
      MAX=80
      GO TO 1
      ENTRY RIN(PROMPT,RVALUE,default, rlow, rhi)
      MODE=1
      CODE=CODE0
      MAX=80
      GO TO 1
      ENTRY IIN(PROMPT,IVALUE,default, ilow, ihi)
      MODE=2
      CODE=CODE0
      MAX=80
      GO TO 1
      ENTRY TIN(PROMPT,TVALUE,default)
      MODE=3
      MAX=LEN(TVALUE)
      GO TO 1
      ENTRY NIN(PROMPT)
      MODE=4
    1 CONTINUE
      CODE=CODE0
      IQ=LEN(PROMPT)
  10  IF(IQ.GT.0 ) then
        if ( mode .ne. 4) then
           WRITE(6,*) PROMPT
        else
c message only; give carriage return and return
          write( 6, *) prompt
          RETURN
        endif
      endif
c  600 FORMAT(A<IQ>$)
c  605 FORMAT(A<IQ>)
      if ( default ) then
	if(mode.eq.0) then
            write(6,604)dvalue
  604	    format(' default: g15.5,' )
        else if(mode.eq.1)then
            write(6,601)rvalue
  601       format('default: g13.3 ' )
        else if( mode .eq. 2)  then
            write(6,602) ivalue
  602       format(' default:  i' )
        else if(mode.eq.3)then
            write(6,603)tvalue
  603       format(' default:  a,' )
        endif

      endif
      READ(5,500) RESPONSE
  500 FORMAT(A100)
      ILOC=1
    2 IF(RESPONSE(1:1).EQ.' ') THEN
            RESPONSE(1:MAX)=RESPONSE(2:MAX)
      ELSE
            GO TO 3
      ENDIF
      ILOC=ILOC+1
      IF(ILOC.GT.MAX) RETURN
      GO TO 2
    3 CONTINUE
      IF(MODE.EQ.3) THEN
            TVALUE=RESPONSE
            RETURN
      ENDIF
      J=INDEX(RESPONSE,' ')-1
      K=16-J
      IF(K.LT.1) K=1
      CODE(K:15)=RESPONSE(1:J)

      if ( mode .eq. 1 ) then
         read( code, 701, err=13) rvalue
         if ( rvalue .lt. rlow .or. rvalue. gt. rhi ) then
           write(6,617)
           write( 6, *)rvalue
           write(6,618)
           write(6,*)rlow,rhi
           code = code0
           goto 10
         endif
      else if ( mode .eq. 0 ) then
         read( code, 701, err=13) dvalue
         if ( dvalue .lt. dlow .or. dvalue. gt. dhi ) then
           write(6,617)
           write( 6, *)dvalue
           write(6,618)
           write(6,*)dlow,dhi
           code = code0
           goto 10
         endif
      else if ( mode .eq. 2 ) then
         read( code, 702, err=13) ivalue
         if ( ivalue .lt. ilow .or. ivalue. gt. ihi ) then
           write( 6, 617)
           write( 6,*) ivalue
           write( 6, 618 )
           write(6, *)ilow,ihi
           code = code0
           goto 10
         endif
      endif
  701 FORMAT(G15.0)
  702 FORMAT(I15)
      RETURN
   13 WRITE(6,613)
  613 FORMAT(' *** Format error: try again ***')
  614 format(' *** Value: g13.3 outside limits: 2g13.3 ***')
  615 format(' *** Value: g15.8 outside limits: 2g15.8 ***')
  617 format(' ****** Value ')
  618 format(' is outside limits: ')
      GO TO 1
      END
c
c
c****************************************************************
c
	logical function yes(prompt,default)
c
	character*(*) prompt
	logical default
	character*1 ans
c

	if(default) then
		ans='Y'
	else
		ans='N'
	endif
c
	call tin(prompt,ans,.false.)
c
	if(ans.eq.'Y'.or.ans.eq.'y') then
		yes=.true.
	else
		yes=.false.
	endif
c
	return
	end
c
c***********************************************************

