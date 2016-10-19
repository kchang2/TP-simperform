
import csv
import pandas as pd
from math import isnan
import os
from sys import exit
import argparse
import numpy as np

def calcRelativeStats(team, average):
    '''
    For a stat, calculates a team's performance relative to the average for that year.
    
    For example, the 2008 Red Sox struck out 1068 times. The average team in 2008 struck out 1099 times. Therefore the Red Sox's relative stat is (100*(1068/1099)) or 97, meaning they struck out 97% as often as the average team. Put another way, the Red Sox struck out 3% less often than the average team.
    '''
    return 100*(float(team) / average)

'''
Note that these are approximate statistics because of the missing SH stat and IBB, which can only be found on the individual players sight. 
However, because we are dealing with JUST understanding the similarities between teams, we use our approximate data to vizualize.

Finally, the appropriate constants in the wOBA varies slightly over time due to run environment. However, for the sake of this project,
we just take the general form.
'''

# CAN INCLUDE /(BPF/100) for BPF

# returns the batting average with balls in play
def calcBABIP(H,HR,AB,K,SF):
    return (float(H)-HR)/(AB-K-HR+SF)

# returns the plate appearances
def calPA(AB,BB,HBP,SF):
    return float(AB)+BB+HBP+SF

# returns the weighted on base average
def calwOBA(BB,HBP,H,secB,thirB,HR,PA):
    firB = H - secB - thirB - HR
    return ((0.75*BB) + (0.9*HBP) + (0.89*firB) + (1.24*secB) + (1.56*thirB) + (1.95 * HR)) / PA

# returns the on base percentage 
def calOBP(BB,H,HBP,AB,SF):
    return (float(H)+BB+HBP)/(AB+BB+HBP+SF)

# Returns the slugging percetnage
def calSLG(H,secB,thirB,HR):
    return float(H) + secB + 2 * thirB + 3 * HR

# Returns the percentage rate for any statistics (ie. K, BB, R, HR)
def NPer(N,AB):
    return float(N)/AB

# Returns the stealing to caught stealing ratio in % form
def calSBCS(SB,CS):
    return float(SB)/CS

# Returns the Walks + Hits per Innings Pitched of the team
def calWHIP(BBA,HA,IPouts):
    return (BBA + HA)/(IPouts/3.00)

# Returns the average ERA of the league
def calERA(ER,IPouts):
    return 9*ER/(IPouts/3.00)

# Returns the percentage rate for any pitching statistics (ie. K/9, BB/9, SO/9)
def calNovr9(N,IPouts):
    return N/(IPouts/3.00)

# Returns the Field Independent Pitching of the team
def calFIP(HRA, BBA, SOA, IPouts):
    return (13*HRA + 3*BBA - 2*SOA) / (IPouts/3.00)

# Returns the adjusted pitchers Batting average for balls in play
def calPBABIP(HA, HRA, IPouts, BBA, SOA, DP, E):
    return (HA - HRA) / ((IPouts + HA + HRA + BBA + SOA) - BBA - HRA + DP/4.00 - E/8.00)

# effective ERA
def caleffERA(ERA, SHO, CG, HR, SV, FP):
    return ERA - 0.018 * SHO - 0.006 * CG + 0.002 * HR + 0.0015 * SV + (1.00 - FP)/2

# Returns the approximate complete games per games played or complete games rate
def calCGRate(CG,IPouts):
    return CG/(IPouts/27.00)

# defines average wins in a 162 game season
def avgWins(games, wins):
    return float(wins)/games * 162

def compareIndivHitting(wOBA1, wOBA2, BABIP1, BABIP2, OBP1, OBP2, SLG1, SLG2, RP1, RP2, KP1, KP2, BBP1, BBP2, HRP1, HRP2, SBCS1, SBCS2):
    '''
    Compares the hitting stats of two teams and calculates how similar the teams are.
    '''
    startingScore = 100
    
	# Offensive stats
    pOffwOBA = wOBA1 - wOBA2
    pOffBABIP = BABIP1 - BABIP2
    pOffOBP = OBP1 - OBP2
    pOffSLG = SLG1 - SLG2
    pOffRP = RP1 - RP2
    pOffKP = KP1 - KP2
    pOffBBP = BBP1 - BBP2
    pOffHRP = HRP1 - HRP2
    pOffSBCS = SBCS1 -SBCS2
    
    totalPointsOff = pOffwOBA + pOffBABIP + pOffOBP + pOffSLG + pOffRP + pOffKP + pOffBBP + pOffHRP + pOffSBCS
    similarityScore = startingScore - totalPointsOff 
    return similarityScore

def compareTeamsPitching(ERA1, ERA2, FIP1, FIP2, WHIP1, WHIP2, K91, K92, BB91, BB92, H91, H92, BABIP1, BABIP2, effERA1, effERA2):
    '''
    Compares the hitting stats of two teams and calculates how similar the teams are.
    '''
    startingScore = 100
    
    # Pitching stats
    pOffERA = ERA1 - ERA2
    pOffBABIP = BABIP1 - BABIP2
    pOffFIP = FIP1 - FIP2
    pOffWHIP = WHIP1 - WHIP2
    pOffK9 = K91 - K92
    pOffBB9 = BB91 - BB92
    pOffH9 = H91 - H92
    pOffEffERA = effERA1 - effERA2
    
    totalPointsOff = pOffERA + pOffBABIP + pOffFIP + pOffWHIP + pOffK9 + pOffBB9 + pOffH9 + pOffEffERA
    similarityScore = startingScore - totalPointsOff 
    return similarityScore

def getTeamInfo(years, teamNames, leagues, gamesPlayed, teamWins, R, AB, H, doubles, triples, HR, BB, SO, SB, CS, HBP, SF, ERA, IPouts, BBA, SOA, HRA, HA, CG, SHO, SV, FP, E, DP, BPF, PPF, ER, avg, index):
    '''
    Returns team's stats relative to the average team of that year. This methodology allows us to fairly compare teams across years by accounting for game-wide changes in offense and defense.
    This method does not take into account the strength of the AL vs. the NL. It just compares each team to all other teams in MLB that year.
    '''
    year = years[index]
    team = teamNames[index]
    games = gamesPlayed[index]
    wins = teamWins[index]
    lg = leagues[index]

    # projected wins in a 162 game season
    projwins = avgWins(games, wins)
    
    #BABIP
    Hhit = H[index]
    HRhit = HR[index]
    ABhit = AB[index]
    SOhit = SO[index]

    sumHhit = int(sums.get_value(year, 'H'))
    sumHRhit = int(sums.get_value(year, 'HR'))
    sumABhit = int(sums.get_value(year, 'AB'))
    sumSOhit = int(sums.get_value(year, 'SO'))

    if isnan(SF[index]): #lahman's DB doesn't have HBP, SF, and CS numbers for many years. if we don't set them to 0, they'll come up as NaN which will screw up the calculations.
        SFhit = 0
    else:
        SFhit = SF[index]
    if isnan(sums.get_value(year, 'SF')):
        sumSFhit = 0
    else:
        sumSFhit = int(sums.get_value(year, 'SF'))

    tBABIP = calcBABIP(Hhit, HRhit, ABhit, SOhit, SFhit)
    lgBABIP = calcBABIP(sumHhit, sumHRhit, sumABhit, sumSOhit, sumSFhit)
    BABIPPlus = calcRelativeStats(tBABIP, lgBABIP)


    #PA
    if isnan(HBP[index]): #lahman's DB doesn't have HBP, SF, and CS numbers for many years. if we don't set them to 0, they'll come up as NaN which will screw up the calculations.
        HBPhit = 0
    else:
        HBPhit = HBP[index]
    if isnan(sums.get_value(year, 'HBP')):
        sumHBPhit = 0
    else:
        sumHBPhit = int(sums.get_value(year, 'HBP'))

    BBhit = BB[index]
    sumBBhit = int(sums.get_value(year, 'BB'))


    tPA = calPA(ABhit, BBhit, HBPhit, SFhit)
    lgPA = calPA(sumABhit, sumBBhit, sumHBPhit, sumSFhit)


    #wOBA
    secBhit = doubles[index]
    thirBhit = triples[index]

    sumsecBhit = int(sums.get_value(year, '2B'))
    sumthirBhit = int(sums.get_value(year, '3B'))


    twOBA = calwOBA(BBhit, HBPhit, Hhit, secBhit, thirBhit, HRhit, tPA)
    lgwOBA = calwOBA(sumBBhit, sumHBPhit, sumHhit, sumsecBhit, sumthirBhit, sumHRhit, lgPA)

    wOBAPlus = calcRelativeStats(twOBA, lgwOBA)


    #OBP
    tOBP = calOBP(BBhit,Hhit,HBPhit,ABhit,SFhit)
    lgOBP = calOBP(sumBBhit,sumHhit,sumHBPhit,sumABhit,sumSFhit)
    OBPPlus = calcRelativeStats(tOBP, lgOBP)


    #SLG
    tSLG = calSLG(Hhit,secBhit,thirBhit,HRhit)
    lgSLG = calSLG(sumHBPhit,sumsecBhit,sumthirBhit,sumHRhit)
    SLGPlus = calcRelativeStats(tSLG,lgSLG)


    #Percent Rate
    Rhit = R[index]

    sumRhit = int(sums.get_value(year, 'R'))

    KPPlus = calcRelativeStats(NPer(SOhit,ABhit), NPer(sumSOhit,sumABhit))
    BBPPlus = calcRelativeStats(NPer(BBhit,ABhit), NPer(sumBBhit,sumABhit))
    RPPlus = calcRelativeStats(NPer(Rhit,ABhit), NPer(sumRhit,sumABhit))
    HRPPlus = calcRelativeStats(NPer(HRhit,ABhit), NPer(sumHRhit,sumABhit))


    #SB/CS ratio
    SBhit = SB[index]
    sumSBhit = int(sums.get_value(year, 'SB'))


    if isnan(CS[index]): #lahman's DB doesn't have HBP, SF, and CS numbers for many years. if we don't set them to 0, they'll come up as NaN which will screw up the calculations.
        CShit = 0
    else:
        CShit = CS[index]
    if isnan(sums.get_value(year, 'CS')):
        sumCShit = 0
    else:
        sumCShit = int(sums.get_value(year, 'CS'))

    tSBCSRate = calSBCS(SBhit,CShit)
    lgSBCSRate = calSBCS(sumSBhit,sumCShit)
    SBCSRatePlus = calcRelativeStats(tSBCSRate,lgSBCSRate)




    #ERA
    ERApitch = ERA[index]
    IPoutspitch = IPouts[index]

    sumERpitch = int(sums.get_value(year, 'ER'))
    sumIPoutspitch = int(sums.get_value(year, 'IPouts'))

    sumERApitch = calERA(sumERpitch,sumIPoutspitch)
    ERAPlus = calcRelativeStats(ERApitch, sumERApitch)


    #WHIP
    BBApitch = BBA[index]
    HApitch = HA[index]
    

    sumBBApitch = int(sums.get_value(year, 'BBA'))
    sumHApitch = int(sums.get_value(year, 'HA'))
    

    tWHIP = calWHIP(BBApitch,HApitch,IPoutspitch)
    lgWHIP = calWHIP(sumBBApitch,sumHApitch,sumIPoutspitch)
    WHIPPlus = calcRelativeStats(tWHIP,lgWHIP)


    #Percent Rate
    SOApitch = SOA[index] 
    sumSOApitch = int(sums.get_value(year, 'SOA'))

    HP9Plus = calcRelativeStats(calNovr9(HApitch,IPoutspitch),calNovr9(sumHApitch,sumIPoutspitch))
    BBP9Plus = calcRelativeStats(calNovr9(BBApitch,IPoutspitch),calNovr9(sumBBApitch,sumIPoutspitch))
    KP9Plus = calcRelativeStats(calNovr9(SOApitch,IPoutspitch),calNovr9(sumSOApitch,sumIPoutspitch))


    #FIP
    HRApitch = HRA[index]
    sumHRApitch = int(sums.get_value(year, 'HRA'))

    tFIP = calFIP(HRApitch,BBApitch,SOApitch,IPoutspitch)
    lgFIP = calFIP(sumHRApitch,sumBBApitch,sumSOApitch,sumIPoutspitch)
    FIPPlus = calcRelativeStats(tFIP,lgFIP)


    #PBABIP
    DPpitch = DP[index]
    Epitch = E[index]

    sumDPpitch = int(sums.get_value(year, 'DP'))
    sumEpitch = int(sums.get_value(year, 'E'))

    tBABIP = calPBABIP(HApitch,HRApitch,IPoutspitch,BBApitch,SOApitch,DPpitch,Epitch)
    lgBABIP = calPBABIP(sumHApitch,sumHRhit,sumIPoutspitch,sumBBApitch,sumSOApitch,sumDPpitch,sumEpitch)
    PBABIPPlus = calcRelativeStats(tBABIP,lgBABIP)


    #effERA
    SHOpitch = SHO[index]
    CGpitch = CG[index]
    SVpitch = SV[index]
    FPpitch = FP[index]
    totalteams = sums.get_value(year, 'yearID') / year

    #average SHO, CG, SV, FP per team
    sumSHOpitch = sums.get_value(year, 'SHO') / totalteams
    sumCGpitch = sums.get_value(year, 'CG') / totalteams
    sumSVpitch =  sums.get_value(year, 'SV') / totalteams
    sumFPpitch = sums.get_value(year, 'FP') / totalteams
    sumHRApitch2 = sumHRApitch / totalteams

    teffERA = caleffERA(ERApitch,SHOpitch,CGpitch,HRApitch,SVpitch,FPpitch)
    lgeffERA = caleffERA(sumERApitch,sumSHOpitch,sumCGpitch,sumHRApitch2,sumSVpitch,sumFPpitch)
    effERAPlus = calcRelativeStats(teffERA,lgeffERA)


    #CG rate
    tCGrate = calCGRate(CGpitch,IPoutspitch)
    lgCGrate = calCGRate(sumCGpitch, sumIPoutspitch)
    CGratePlus = calcRelativeStats(tCGrate,lgCGrate)


    return year, team, lg, projwins, BABIPPlus, wOBAPlus, OBPPlus, SLGPlus, KPPlus, BBPPlus, RPPlus, HRPPlus, SBCSRatePlus, ERAPlus, WHIPPlus, HP9Plus, BBP9Plus, KP9Plus, FIPPlus, PBABIPPlus, effERAPlus, CGratePlus


def readDatabase(datafile):
    '''
    Read data from the Lahman database file.
    
    Note that I had to rename the 2B and 3B columns. Python doesn't like identifiers that start with numbers.
    '''
    try:
        df = pd.read_csv(dataFile)
       
        #Get teams' info
        years = df["yearID"]
        numSeasons = len(years) #2805 teams within total season
        teamNames = df["name"]
        league = df['lgID']
        gamesPlayed = df["G"]
        teamWins = df["W"]
        
        
        #Get stats for offense
        R = df["R"]
        AB = df["AB"] # at bats by offense
        H = df["H"] #hits by offense
        doubles = df["2B"]
        triples = df["3B"]
        HR = df["HR"] # home runs by offense
        BB = df["BB"] # walks by offense
        SO = df["SO"] # strikeouts by offense
        SB = df["SB"] # stolen bases
        CS = df["CS"] # caught stealing
        HBP = df["HBP"] # offense hit by pitch
        SF = df["SF"] # sacrifice flies by offense
        
        #Get stats for pitchers
        ERA = df["ERA"]
        ER = df["ER"]
        IPouts = df["IPouts"] # Outs Pitched by pitchers (innings pitched x 3)
        BBA = df["BBA"] # walks allowed by pitchers
        SOA = df["SOA"] # strikeouts by pitchers
        HRA = df["HRA"] # home runs allowed by pitchers
        HA = df["HA"] # hits allowed by pitchers
        CG = df["CG"] # complete games pitched
        SHO = df["SHO"] # shutouts
        SV = df["SV"] # saves
        FP = df["FP"] # fielding percentage
        E = df["E"] # team errors
        DP = df["DP"]

        BPF = df["BPF"] # Three-year park factor for batters
        PPF = df["PPF"] # Three-year park factor for pitchers
        
        #Compute annual averages for all the stats we care about
        grouped = df.groupby('yearID')
        avg = grouped.agg({'HBP': np.mean, 'SF': np.mean, 'ERA': np.mean, 'ER': np.mean, 'IPouts': np.mean, 'SV': np.mean, 'FP': np.mean, 'E': np.mean, 'BB': np.mean, 'SO': np.mean, 'R':np.mean, 'AB': np.mean, 'RA': np.mean, 'HR': np.mean, 'BBA': np.mean, 'SOA': np.mean, 'HRA': np.mean, 'SB': np.mean, 'CS': np.mean, 'E': np.mean, 'DP': np.mean, '2B': np.mean, '3B': np.mean, 'H': np.mean, 'HA': np.mean, 'CG': np.mean, 'SHO': np.mean})
        sums = grouped.aggregate({'yearID': np.sum,'HBP': np.sum, 'SF': np.sum, 'ERA': np.sum, 'ER': np.sum, 'IPouts': np.sum, 'SV': np.sum, 'FP': np.sum, 'E': np.sum, 'BB': np.sum, 'SO': np.sum, 'R':np.sum, 'AB': np.sum, 'RA': np.sum, 'HR': np.sum, 'BBA': np.sum, 'SOA': np.sum, 'HRA': np.sum, 'SB': np.sum, 'CS': np.sum, 'E': np.sum, 'DP': np.sum, '2B': np.sum, '3B': np.sum, 'H': np.sum, 'HA': np.sum, 'CG': np.sum, 'SHO': np.sum, 'G': np.sum})
        return years, numSeasons, teamNames, league, gamesPlayed, teamWins, R, AB, H, doubles, triples, HR, BB, SO, SB, CS, HBP, SF, ERA, IPouts, BBA, SOA, HRA, HA, CG, SHO, SV, FP, E, DP, BPF, PPF, ER, avg, sums
    except Exception as e:
        exit("Error reading from %s: %s" % (dataFile, e))

def createOutputFiles(years, teamNames, numSeasons):
    '''
    Creates the .csv files that will hold the results of comparing each team to the rest.
    '''
    for i in range (2686, numSeasons):
        year = years[i]
        teamName = teamNames[i]
        if teamName == "Chicago/Pittsburgh (Union League)":
            # the / causes an error because the computer thinks it's a dir separator 
            teamName = "Chicago Pittsburgh (Union League)"
            teamNames.loc[i] = teamName #replace modified team name in array
        teamSeasonFile = str(year) + " " + teamName + '.csv'
        resultFile = os.path.join(dir, teamSeasonFile)
        try:
            f = open(resultFile,'w')
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()
        except Exception as e:
            print "Error creating %s: %s" % (resultFile, e)


parser = argparse.ArgumentParser(description='Compares baseball teams throughout history according to how well they performed relative to league average.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                   default=False,
                   help='Enable verbose mode.')
args = parser.parse_args()

dataFile = "/Users/kaichang/Documents/classes/ay119/final_project/lahman-csv/Teams.csv"
years, numSeasons, teamNames, league, gamesPlayed, teamWins, R, AB, H, doubles, triples, HR, BB, SO, SB, CS, HBP, SF, ERA, IPouts, BBA, SOA, HRA, HA, CG, SHO, SV, FP, E, DP, BPF, PPF, ER, avg, sums = readDatabase(dataFile)
header = ["comparedTeam", "League", "Wins", "SimiliartyHitting", "SimilarityPitching","BABIP","wOBA","ERA","FIP","PBABIP","effERA"]
dir = "results/"
if not os.path.exists(dir): # create results directory if it doesn't exist already 
    os.makedirs(dir)
createOutputFiles(years, teamNames, numSeasons)

# compare teams, calculate scores, and write the scores to a file    
# starts at 2012
for j in range (2686, numSeasons):
        year1, team1, lg1, projwins1, BABIPPlus1, wOBAPlus1, OBPPlus1, SLGPlus1, KPPlus1, BBPPlus1, RPPlus1, HRPPlus1, SBCSRatePlus1, ERAPlus1, WHIPPlus1, HP9Plus1, BBP9Plus1, KP9Plus1, FIPPlus1, PBABIPPlus1, effERAPlus1, CGratePlus1 = getTeamInfo(years, teamNames, league, gamesPlayed, teamWins, R, AB, H, doubles, triples, HR, BB, SO, SB, CS, HBP, SF, ERA, IPouts, BBA, SOA, HRA, HA, CG, SHO, SV, FP, E, DP, BPF, PPF, ER, avg, j)
        id1 = str(year1) + ' ' + team1
        if args.verbose:
            print "Comparison team: %s" %id1
        fileToOpen = os.path.join(dir, id1) + '.csv'
        try:
            # Open Team J's results file for writing
            f = open(fileToOpen, 'a')
            results = csv.writer(f)
            row = [id1, lg1, projwins1, 100, 100, BABIPPlus1, wOBAPlus1, ERAPlus1, FIPPlus1, PBABIPPlus1, effERAPlus1]
            results.writerow(row)
            #runs and checks similarities dating back to 1960
            for k in range (1344, numSeasons):
                year2, team2, lg2, projwins2, BABIPPlus2, wOBAPlus2, OBPPlus2, SLGPlus2, KPPlus2, BBPPlus2, RPPlus2, HRPPlus2, SBCSRatePlus2, ERAPlus2, WHIPPlus2, HP9Plus2, BBP9Plus2, KP9Plus2, FIPPlus2, PBABIPPlus2, effERAPlus2, CGratePlus2 = getTeamInfo(years, teamNames, league, gamesPlayed, teamWins, R, AB, H, doubles, triples, HR, BB, SO, SB, CS, HBP, SF, ERA, IPouts, BBA, SOA, HRA, HA, CG, SHO, SV, FP, E, DP, BPF, PPF, ER, avg, k)
                id2 = str(year2) + ' ' + team2                
                if (id1 != id2): # prevent comparing a team to itself
                    row = [] # start a blank row for a new comparison
                    row.append(id2) #add the comparison team as the first column
                    row.append(lg2)
                    row.append(projwins2)
                    simScoreHit = compareIndivHitting(wOBAPlus1, wOBAPlus2, BABIPPlus1, BABIPPlus2, OBPPlus1, OBPPlus2, SLGPlus1, SLGPlus2, RPPlus1, RPPlus2, KPPlus1, KPPlus2, BBPPlus1, BBPPlus2, HRPPlus1, HRPPlus2, SBCSRatePlus1, SBCSRatePlus2)
                    row.append(simScoreHit)
                    simScorePitch = compareTeamsPitching(ERAPlus1, ERAPlus2, FIPPlus1, FIPPlus2, WHIPPlus1, WHIPPlus2, KP9Plus1, KP9Plus2, BBP9Plus1, BBP9Plus2, HP9Plus1, HP9Plus2, PBABIPPlus1, PBABIPPlus2, effERAPlus1, effERAPlus2)
                    row.extend((simScorePitch,BABIPPlus2,wOBAPlus2,ERAPlus2,FIPPlus2,PBABIPPlus2,effERAPlus2))
                    results.writerow(row)
        except Exception as e:
            print "Error opening %s: %s" % (fileToOpen, e) 
        f.close() #we are done with team J's CSV file.       
