import numpy as np

# iGSE for calculating inductor core loss

# minorLoop function
def minorLoop1(s, p):
    qr = len(s)
    peak = -1
    prevSlope = 0
    for i in range(1, qr):
        slope = (s[i - 1] - s[i]) / (p[i - 1] - p[i])
        # this should be elif definitely and probably not >=, breaks when peak is flat
        if slope < 0 and prevSlope >= 0:
            peak += 1
        elif prevSlope <= 0 and slope > 0:
            peak += 1
        prevSlope = slope

    if peak > 2:
        m1 = 1
    else:
        m1 = 0

    return m1


# end minorLoop1 function

##########################################################################################
# Convert a PWL waveform w (must be a vector) represented by the points w at times t     #
# to a PWL waveform with points at any zero crossings, thus any segment of the returned  #
# waveform is all positive or all negative                                               #
##########################################################################################
def makePositive(t, w):
    # Different than way it's done in MATLAB - check
    i = 0
    while i < len(w) - 1:
        if w[i] * w[i + 1] < 0:
            tcross = w[i] * (t[i + 1] - t[i]) / (w[i] - w[i + 1]) + t[i]
            # provision for iff the increment is negligible
            if tcross == t[i]:
                w[i] = 0
            elif tcross == t[i+1]:
                w[i+1] = 0
            else:
                t.insert(i + 1, tcross)
                w.insert(i + 1, 0)
        i += 1

    w = [abs(i) for i in w]

    return [t, w]


# end makePositive function

# Calculate loss for loop segment using improved equation
def calcseg(t, B, alpha, beta, k1, a):
    # a is a fraction of Bpp used, 1-a is a graction of original
    bma1a = (beta - alpha) * (1 - a)  # exponent of B(t)
    bmaa = (beta - alpha) * a  # Exponent of Bpp

    Bpp = max(B) - min(B)

    if any(n < 0 for n in B):
        [t, B] = makePositive(t, B)

    length = len(t)
    T = t[len(t) - 1] - t[0]

    deltaB = []
    deltat = []

    for i in range(1, len(B)):
        deltaB.append(abs(B[i] - B[i - 1]))
        deltat.append(t[i] - t[i - 1])

    dBdt = []

    for i in range(len(deltaB)):
        if deltat[i] == 0:
            pass
        dBdt.append(deltaB[i] / deltat[i])

    m1 = [i ** (alpha - 1) for i in dBdt]
    m2 = []

    for i in range(1, len(B)):
        m2.append(abs((B[i]) ** (bma1a + 1) - B[i - 1] ** (bma1a + 1)))

    pseg = 0
    for i in range(0, len(m1)):
        pseg += m1[i] * m2[i]
    pseg = k1 / T / (bma1a + 1) * pseg * Bpp ** bmaa

    return pseg


# End calcseg function


# This function takes a PWL current waveform corresponding to a B-H loop
# Splits waveform into smaller waveforms with same starting and ending values
# Inputs: a -> B vector corresponding to B-H loop, b time vector of currents that correspond to different minor B-H loop
# Outputs: B values and time vector corresponding to major B-H loop, same for minor B-H loop
def splitloop(a, b):
    ##########################################################
    # Reshaping the waveforms and identifying the peak point #
    ##########################################################

    # Identify index of lowest point in a
    e = a.index(min(a))

    # v is vector which stores shifted values of a
    v = []
    # t is vector which stores shifted values of b
    t = []

    # Adds lowest point for first value (time zero)
    v.append(a[e])
    t.append(0)

    # Adjusts for the times corresponding to the values
    bdiff = np.diff(b, axis=0)
    cumdiff = 0

    # Loop stores all value of a from lowest point to endpoint in v
    for i in range(e + 1, len(a)):
        v.append(a[i])  # Values of a being stored in v
        cumdiff += bdiff[i - 1]  # Time adjustment
        t.append(cumdiff)

    # Loop stores all values of a from starting point to lowest point in v
    for j in range(1, e + 1):  # Appends the last part before the lowest point
        v.append(a[j])
        cumdiff += bdiff[j - 1]
        t.append(cumdiff)

    # Finds the position of the peak value
    z = v.index(max(v))

    #################################################################
    # End of reshaping the waveforms and identifying the peak point #
    #################################################################

    # Defining variables to keep track of values in the vectors
    # reduced by one from MATLAB algorithm because Python lists start at 0
    i = 1
    j = 0
    k = 0

    s = [[]]  # Defining cells for minorloop
    p = [[]]  # Defining cells for minortime

    # m stores the major loop values extracted from v
    m = []
    # n stores the corresponding time values extracted from t
    n = []

    #############################################
    # Splits the rising portion of the waveform #
    #############################################

    m.append(v[0])
    n.append(t[0])

    while i <= z:  # Checks all values before peak
        if v[i] >= v[i - 1]:  # Compares adjacent values of v to see if waveform is rising
            m.append(v[i])  # Add an element of v to m
            n.append(t[i])  # Add an element of t to n
            count = 1  # Counter to keep track of end of rising part

        else:
            # Check for minor loop in rising part
            s[j].append(v[i - 1])  # Add an element of v to s
            p[j].append(t[i - 1])  # Add an element of t to p
            k += 1

            # Repeat process until minor loop ends
            while v[i] < max(m):
                s[j].append(v[i])
                p[j].append(t[i])
                k += 1
                i += 1

            # Calculate the slope of the rising edge of the minor loop
            slope = (v[i - 1] - v[i]) / (t[i - 1] - t[i])

            # Makes the last element of the minor loop same as the mximum value of m
            s[j].append(max(m))

            # Computes the value of time of the point which was stored last in s
            stemp = ((max(m) - v[i - 1]) / slope) + t[i - 1]
            p[j].append(stemp)  # Value is stored in p
            m.append(max(m))  # The last point in m is repeated for continuity
            n.append(stemp)  # The time value is also repeated
            count += 1  # Counter is incremented to indicate end of minor loop
            j += 1  # Index of s is incremented
            # Add empty arrays to s and p each time j is incremented
            p.append([])
            s.append([])

            k = 0

        if count <= 1:  # Check condition keeping track of end of rising part
            i += 1  # Increment counter keeping track of the elements in v

    ##########################################################
    # End of splitting of the rising portion of the waveform #
    ##########################################################

    # m now stores the rising part of the major loop
    # n now stores the corresponding time values of the major loop
    # s stores the minor loops in the rising part of the waveform
    # p stores the corresponding time values of the minor loop

    ##############################################
    # Splits the falling portion of the waveform #
    ##############################################
    while i < len(v):  # Checks for all values after the peak point
        if v[i] <= v[i - 1]:
            m.append(v[i])  # Add element of v to m
            n.append(t[i])  # Add element of t to n
            count = 1  # Counter to keep track of end of rising part

        else:  # Check for minor loop in the falling part
            temp = v[i - 1]  # Temporarily store last value in v
            s[j].append(v[i - 1])  # Add element of v to s
            p[j].append(t[i - 1])  # Add element of t to p
            k += 1

            # Compares adjacent values of v for all the remaining v values
            while (i < len(v) and v[i] > temp):
                s[j].append(v[i])  # Adds an element of v to s
                p[j].append(t[i])  # Adds an element of t to p
                k += 1
                i += 1

            while i < len(v) and k > 1:
                # Calculate slope of rising edge of minor loop
                slope = (v[i - 1] - v[i]) / (t[i - 1] - t[i])

                # Makes the first element of the minor loop same as the last element in m
                s[j].append(temp)

                # Computes the value of time at the point last stored in s
                qtemp = ((temp - v[i - 1]) / slope) + t[i - 1]  # Calculating last time value
                p[j].append(qtemp)  # Value is stored in p
                r = 1  # Counter to make the pass through the loop only once
                while v[i] != temp and r == 1:
                    m.append(temp)  # Last point in m repeated for continuity
                    n.append(qtemp)  # The time value is also repeated
                    r += 1  # Counter incremented to indicate the pass has been made

                count += 1  # Count is incremented to indicate end of the minor loop

                j += 1  # Index of s is incremented
                # Add empty arrays to s and p each time j is incremented
                p.append([])
                s.append([])

                k = 1
        if count <= 1:  # Check condition keeping track of end of falling part
            i += 1  # Increment counter keeping track of the elements in v

    ###########################################################
    # End of splitting of the falling portion of the waveform #
    ###########################################################

    #############################################################
    # Removal of repetition of points in m and adjusting values #
    #############################################################
    x = len(m)  # Variable stores length of m
    majorLoop = [m[0]]  # Stores first element of m
    majorTime = [n[0]]  # Stores the first element of n
    g = 0  # Initializes variable which adjusts time for points following the point of repetition

    for h in range(1, x):
        if m[h - 1] != m[h]:
            majorTime.append(n[h] - g)
            majorLoop.append(m[h])
        else:
            g += n[h] - n[h - 1]

    ####################################################################
    # End of removal of repetition of points in m and adjusting values #
    ####################################################################

    #############################################################################
    # Finding the number of minor loops to be used for checking sub loops later #
    #############################################################################

    uo = 0
    pt = 0  # Variable to keep track of minor loops
    ss = 1

    while ss == 1 and uo < len(s):
        while len(s[uo]) > 0:
            uo += 1
        ss += 1
        # Original code has pt = uo-1 but we start uo at 0 not 1 because of Python indices
        pt = uo

    ######################################################################
    # Recursion checks if any portion of the split waveform has subloops #
    # If so, repeats above process to make it a single loop              #
    ######################################################################

    minorLoop = []
    minorTime = []

    for qq in range(0, pt):
        sinp = s[qq]  # flux
        pinp = p[qq]  # time

        foo = minorLoop1(sinp, pinp)
        if foo:
            fn, ln, sn, pn = splitloop(sinp, pinp)
            if fn and sn:
                minorLoop.append(fn)
                minorLoop.append(sn[0])
                minorTime.append(ln)
                minorTime.append(pn[0])
        elif sinp:
            minorLoop.append(sinp)
            minorTime.append(pinp)

    # left off at coreloss.m Line #428
    return [majorLoop, majorTime, minorLoop, minorTime]


# end splitLoop function

# Function to calculate core loss per unit volume using GSE for PWL signal form
def gsepwl(t, B, alpha, beta, k):
    # a=1 for iGSE
    # a = 0 for GSE
    a = 1.0  # a is fraction of Bpp used, 1-a is fraction of original

    # Total time of PWL period
    T = t[len(t) - 1] - t[0]
    ki = k / ((2 ** (beta + 1)) * (np.pi ** (alpha - 1)) * (.2761 + 1.7061 / (alpha + 1.354)))

    # Split waveform into major and minor loops
    B, t, Bs, ts = splitloop(B, t)  # Split waveform into major and minor loops

    pseg = []
    dt = []

    pseg.append(calcseg(t, B, alpha, beta, ki, a))
    dt.append(t[len(t) - 1] - t[0])

    for j in range(0, len(ts)):
        pseg.append(calcseg(ts[j], Bs[j], alpha, beta, ki, a))
        tseg = ts[j]
        dt.append(tseg[len(tseg) - 1] - tseg[0])

    p = 0
    for i in range(0, len(pseg)):
        p += (pseg[i] * dt[i]) / T

    return p


# End gsepwl function

#########################################################################################
# time and B are a piecewise linear series of data points                               #
# time is a vector of successive time values corresponding to flux values, in seconds   #
# B = vector of flux densities corresponding to time vector values, in Tesla            #
# alpha, beta, k are Steinmetz parameters, must be used with MKS units                  #
# SuppreessFlag dictates how much data to show                                          #
# coreloss will calculate the power loss in your specified ferrite core in W/m^3        #
#########################################################################################
def coreloss(tvs, B, alpha, beta, k):

    # Error checks
    error = "None"
    suppressFlag = 0

    for i in range(1, len(tvs)):
        if tvs[i - 1] >= tvs[i]:
            error = "Time data points must be successive"

    if len(tvs) != len(B):
        error = 'Time and flux vectors must have same length'

    # if B[0] != B[len(B) - 1]:
    #     error = 'Since the PWL input is periodic, the first flux value must equal the last'
    # Error checks done

    # Calculate core loss if there is no error with the input data
    if not suppressFlag:

        if error == "None":
            # tvs as seen in original code is complex conjugate of matrix tvs
            Pcore = 1000 * gsepwl(np.conjugate(tvs), B, alpha, beta, k)
            # print("Core Loss:", Pcore, "[W/m^3]")
            y = Pcore

        else:
            print("Error:", error)
            y = -1
    elif suppressFlag:  # Supress Output
        if error == 'None':
            y = 1000 * gsepwl(np.conjugate(tvs), B, alpha, beta, k)
        else:
            y = -1  # Error Occurred

    # print("iGSE [W/m^3]: ", y)

    return y
# End coreloss function