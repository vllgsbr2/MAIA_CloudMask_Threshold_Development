C FILE: svi_calculation.F
C USEAGE:
C       f2py -m svi_calculation -c svi_calculation.F
C TO GENERATE PYTHON USABLE .SO FILE

      SUBROUTINE svi_calculation(RADS, VAL_BAD_DATA, NUMBER_MIN_VALID,
     & SVIS)

      IMPLICIT NONE

C     GRANULE RADIANCE DATA & BAD DATA FLAG
      REAL, DIMENSION(1000, 1000) :: RADS
      REAL VAL_BAD_DATA

C     MINIMUM NUMBER OF VALID RADIANCE DATA WITHIN A 3X3 REGION
      INTEGER NUMBER_MIN_VALID

C     OUTPUT ARRAY
      REAL, DIMENSION(1000, 1000) :: SVIS

C     INTERNAL USE INDEXES
      INTEGER I, J
      REAL, DIMENSION(9) :: TMP_RAD
      REAL, EXTERNAL :: H_SIGMA


Cf2py intent(in) rads, val_bad_data, number_min_valid
Cf2py intent(out) svis


      SVIS = 0.0

      DO I = 1, 1000
        DO J = 1, 1000

C     LEFT-TOP CORNER
          IF (I.EQ.1 .AND. J.EQ.1) THEN
            TMP_RAD(1:4) = RESHAPE(RADS(I:I+1, J:J+1), (/4/))
            SVIS(I, J) = H_SIGMA(TMP_RAD, VAL_BAD_DATA,
     &                           NUMBER_MIN_VALID, 4)
C     RIGHT-BOTTOM CORNER
          ELSE IF (I.EQ.1000 .AND. J.EQ.1000) THEN
            TMP_RAD(1:4) = RESHAPE(RADS(I-1:I, J-1:J), (/4/))
            SVIS(I, J) = H_SIGMA(TMP_RAD, VAL_BAD_DATA,
     &                           NUMBER_MIN_VALID, 4)
C     RIGHT-TOP CORNER
          ELSE IF (I.EQ.1 .AND. J.EQ.1000) THEN
            TMP_RAD(1:4) = RESHAPE(RADS(I:I+1, J-1:J), (/4/))
            SVIS(I, J) = H_SIGMA(TMP_RAD, VAL_BAD_DATA,
     &                           NUMBER_MIN_VALID, 4)
C     LEFT-BOTTOM CORNER
          ELSE IF (I.EQ.1000 .AND. J.EQ.1) THEN
            TMP_RAD(1:4) = RESHAPE(RADS(I-1:I, J:J+1), (/4/))
            SVIS(I, J) = H_SIGMA(TMP_RAD, VAL_BAD_DATA,
     &                           NUMBER_MIN_VALID, 4)
C     OTHER TOP LINE
          ELSE IF (I.EQ.1) THEN
            TMP_RAD(1:6)= RESHAPE(RADS(I:I+1, J-1:J+1), (/6/))
            SVIS(I, J) = H_SIGMA(TMP_RAD, VAL_BAD_DATA,
     &                           NUMBER_MIN_VALID, 6)
C     OTHER BOTTOM LINE
          ELSE IF (I.EQ.1000) THEN
            TMP_RAD(1:6) = RESHAPE(RADS(I-1:I, J-1:J+1), (/6/))
            SVIS(I, J) = H_SIGMA(TMP_RAD, VAL_BAD_DATA,
     &                           NUMBER_MIN_VALID, 6)
C     OTHER LEFT COLUMN
          ELSE IF (J.EQ.1) THEN
            TMP_RAD(1:6) = RESHAPE(RADS(I-1:I+1, J:J+1), (/6/))
            SVIS(I, J) = H_SIGMA(TMP_RAD, VAL_BAD_DATA,
     &                           NUMBER_MIN_VALID, 6)
C     OTHER RIGHT COLUMN
          ELSE IF (J.EQ.1000) THEN
            TMP_RAD(1:6) = RESHAPE(RADS(I-1:I+1, J-1:J), (/6/))
            SVIS(I, J) = H_SIGMA(TMP_RAD, VAL_BAD_DATA,
     &                           NUMBER_MIN_VALID, 6)
C     THE REST PIXELS
          ELSE
            TMP_RAD = RESHAPE(RADS(I-1:I+1, J-1:J+1), (/9/))
            SVIS(I, J) = H_SIGMA(TMP_RAD, VAL_BAD_DATA,
     &                           NUMBER_MIN_VALID, 9)

          ENDIF
        ENDDO
      ENDDO

      END
C END FILE SAMPLE2GRID_SW.F


      FUNCTION H_SIGMA(ARRAY_1D, VAL_BAD_DATA, NUMBER_MIN_VALID,
     &   NUMBER_DATA)

        IMPLICIT NONE
        REAL, DIMENSION(9) :: ARRAY_1D
        REAL H_SIGMA, VAL_BAD_DATA
        INTEGER NUMBER_MIN_VALID, NUMBER_DATA
C       INTERNAL USE VARIABLES
        REAL TMP_SUM, TMP_MEAN
        INTEGER I, N

        TMP_SUM = 0.0
        N = 0

C       CALCULATE SUM OF THE VALID RADIANCES AND COUNT
        DO I = 1, NUMBER_DATA
          IF (ARRAY_1D(I) .NE. VAL_BAD_DATA) THEN
            TMP_SUM = TMP_SUM + ARRAY_1D(I)
            N = N + 1
          ENDIF
        ENDDO
C       ONLY CALCULATE H_SIGMA WHEN THE NUMBER OF VALID RADIANCES
C       IS LARGER THAN THE GIVEN THRESHOLD
        IF (N .GE. NUMBER_MIN_VALID) THEN
          TMP_MEAN = TMP_SUM / N
          TMP_SUM = 0.0
          DO I = 1, NUMBER_DATA
            IF (ARRAY_1D(I) .NE. VAL_BAD_DATA) THEN
              TMP_SUM = TMP_SUM + (ARRAY_1D(I) - TMP_MEAN)**2
            ENDIF
          ENDDO
          H_SIGMA = SQRT(TMP_SUM/N)
C       OTHERWISE SET THE H_SIGMA -999.
        ELSE
          H_SIGMA = -999.
        ENDIF
      END

