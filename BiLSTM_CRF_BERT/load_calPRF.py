def readfiles():
  filenames = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/result_text.txt"
  d = {}
  true_ent = []
  count = 0
  test_result = {}
  with open(filenames,"r")as f:
    for i in f.readlines():
      print(i.split(","))
      test_result[count] = i.split(",")
      count+=1
  print(test_result)


true_result = {0: ['Application', 'Analytical', 'Solution', 'Screening', 'Tool', 'Sea Water', 'Intrusion', 'Sea water', 'intrusion', 'aquifers', 'problematic', 'coastal areas', 'physics', 'chemistry', 'complex', 'sea water', 'intrusion', 'quantify', 'assessment tools', 'analytical models', 'application', 'study', 'sharp-interface analytical approach', 'estimating', 'sea water', 'homogeneous', 'coastal', 'aquifer', 'pumping', 'regional flow', 'effects', 'steady-state conditions', 'analytical model', 'tested', 'observations', 'Canada', 'United States', 'Australia', 'assess', 'initial', 'approximation', 'sea water', 'groundwater', 'management', 'decision making', 'occurrence', 'sea water', 'intrusion', 'increased', 'salinity', 'pumping wells', 'approximately', 'cases', 'Application', 'correction', 'dispersion', 'improve', 'results', 'Failure', 'analytical model', 'correct', 'predictions', 'mismatches', 'results', 'salt water', 'wedge', 'coast', 'predevelopment conditions', 'Predictions', 'aquifers', 'salt water', 'wedge', 'inland', 'predevelopment conditions', 'pumping', 'Sharp-interface solutions', 'tools', 'screen', 'vulnerability', 'coastal', 'aquifers', 'sea water', 'intrusion', 'study', 'sharp-interface results'], 1: ['Patient', 'Physician', 'Discordance', 'Global Assessment', 'Rheumatoid Arthritis', 'Systematic', 'Literature Review', 'Meta-Analysis', 'patient', 'therapeutic', 'decision-making', 'management', 'rheumatoid arthritis', 'RA', 'patient', 'opinion', 'disease', 'status', "physician's", 'opinion', 'study', 'assess', 'published literature', 'frequency', 'patient', 'physician', 'discordance', 'global assessment', 'RA', 'systematic', 'literature review', 'articles', 'published', 'Medline', 'Embase', 'discordance', 'RA', 'investigators', 'Discordance', 'absolute difference', 'patient global', 'PGA', 'physician global assessments', 'PhGA', 'frequency', 'discordance', 'predictors', 'study', 'Frequencies', 'discordance', 'pooled', 'meta-analysis', 'random', 'effect', 'studies', 'patients', 'weighted mean', 'age', 'years', 'weighted mean', 'disease duration', 'years', 'women', 'value', 'difference', 'PGA', 'PhGA', 'discordance', 'weighted mean value', 'pooled', 'percentage', 'patients', 'discordance', 'confidence interval', 'range', 'PGA', 'higher', 'PhGA', 'PGA', 'pain', 'functional', 'incapacity', 'PhGA', 'joint counts', 'acute-phase reactants', 'Discordance', 'global assessment', 'frequently', 'difference', 'half', 'patients', 'discordant', 'long-term', 'consequences', 'discordance']}
test_result = {0: ['Application', 'Analytical Solution', 'Screening Tool', 'Sea Water', 'Intrusion', 'Sea water', 'intrusion', 'aquifers', 'coastal areas', 'physics', 'chemistry', 'issue', 'complex', 'sea water', 'intrusion', 'quantify', 'Simple assessment tools', 'analytical models', 'application', 'applicability', 'field situations', 'unclear', ' study', ' reliability', ' sharp-interface analytical approach', ' estimating', ' extent', ' sea water', ' homogeneous', ' coastal aquifer', ' pumping', ' regional', ' effects', ' steady-state conditions', ' analytical model', ' tested', ' observations', ' Canada', ' United States', ' Australia', ' approximation', ' sea water', ' extent', ' groundwater', ' management', ' decision making', ' occurrence', ' sea water', ' intrusion', ' increased', ' salinity', ' pumping wells', ' cases', ' correction', ' dispersion', ' improve', ' results', ' Failure', ' analytical model', ' predictions', ' mismatches', ' assumptions', ' complex field settings', ' results', ' salt water', ' coast', ' predevelopment conditions', ' Predictions', ' poorest', ' aquifers', ' salt water', ' inland', ' predevelopment conditions', ' dispersive', ' pumping', ' Sharp-interface solutions', ' tools', ' screen', ' vulnerability', ' coastal aquifers', ' sea water', ' intrusion', ' sources', ' uncertainty', ' identified', ' study', ' sharp-interface results\n'], 1: ['Patient', ' Physician', ' Discordance', ' Global Assessment', ' Rheumatoid Arthritis', ' Systematic Literature Review', ' Meta-Analysis\n'], 2: ['Nosocomial pneumonia', ' methicillin-resistant Staphylococcus aureus', ' treated with', ' linezolid', ' vancomycin', ' secondary', ' resource use', ' Spanish', ' perspective', ' Spanish', ' perspective', ' study', ' assess', ' healthcare resource utilization', ' HCRU', ' costs', ' treating', ' nosocomial pneumonia', ' NP', ' methicillin-resistant Staphylococcus aureus', ' MRSA', ' hospitalized adults', ' linezolid', ' vancomycin', ' evaluation', ' renal failure', ' rate', ' economic outcomes', ' study groups', ' economic post hoc', ' evaluation', ' randomized', ' double-blind', ' multicenter phase 4 study', ' Nosocomial pneumonia', ' MRSA', ' hospitalized adults', ' modified', ' intent', ' treat', ' mITT', ' population', ' linezolid', ' vancomycin', ' treated', ' patients', ' Costs', ' HCRU', ' evaluated', ' patients', ' administered', ' linezolid', ' vancomycin', ' patients', ' renal failure', ' Analysis', ' HCRU', ' outcomes', ' costs', ' Total costs', ' linezolid', ' vancomycin', ' treated', ' patients', ' renal failure', ' rate', ' linezolid', ' treated', ' patients', ' higher', ' patients', ' renal failure', ' patients', ' renal failure', ' HCRU', ' days', ' mechanical ventilation', ' days', ' ICU stay', ' days', ' hospital stay', ' days', ' cost', ' linezolid', ' vancomycin', ' treated', ' patients', ' statistically significant', ' costs', ' patient', ' day', ' cohorts', ' mortality', ' Spanish', ' perspective', ' statistically significant', ' differences', ' linezolid', ' vancomycin', ' pneumonia', ' cohorts', ' drug cost', ' linezolid', ' renal failure', ' adverse events\n']}
test_result2 ={0: ['Application', ' Analytical Solution', ' Screening Tool', ' Sea Water', ' Intrusion', ' Sea water', ' intrusion', ' aquifers', ' problematic', ' coastal areas', ' physics', ' chemistry', ' issue', ' complex', ' sea water', ' intrusion', ' quantify', ' assessment', ' analytical models', ' application', ' applicability', ' field situations', ' study', ' reliability', ' sharp-interface analytical approach', ' estimating', ' sea water', ' homogeneous', ' coastal aquifer', ' pumping', ' regional', ' flow effects', ' steady-state conditions', ' analytical model', ' observations', ' Canada', ' United States', ' Australia', ' approximation', ' sea water', ' groundwater', ' management', ' decision making', ' occurrence', ' sea water', ' intrusion', ' increased', ' salinity', ' pumping wells', ' predicted', ' cases', ' correction', ' dispersion', ' improve', ' results', ' Failure', ' analytical model', ' predictions', ' simplifying', ' assumptions', ' complex field settings', ' results', ' toe', ' salt water', ' wedge', ' coast', ' predevelopment conditions', ' Predictions', ' aquifers', ' salt water', ' wedge', ' predevelopment conditions', ' dispersive', ' pumping', ' Sharp-interface solutions', ' tools', ' screen', ' vulnerability', ' coastal', ' sea water', ' intrusion', ' significant', ' sources', ' uncertainty', ' identified', ' study', ' sharp-interface results)\n']}
test_result3 = {0: ['Application', ' Analytical Solution', ' Screening Tool', ' Sea Water', ' Intrusion', ' Sea water', ' intrusion', ' aquifers', ' coastal areas', ' physics', ' chemistry', ' issue', ' complex', ' sea water', ' intrusion', ' quantify', ' analytical models', ' applicability', ' field situations', ' study', ' reliability', ' sharp-interface analytical approach', ' estimating', ' sea water', ' homogeneous', ' coastal', ' aquifer', ' pumping', ' regional', ' flow effects', ' steady-state conditions', ' analytical model', ' tested', ' observations', ' Canada', ' United States', ' Australia', ' assess', ' approximation', ' sea water', ' groundwater', ' management', ' decision making', ' occurrence', ' sea water', ' intrusion', ' increased', ' salinity', ' pumping wells', ' cases', ' Application', ' correction', ' dispersion', ' improve', ' results', ' Failure', ' analytical model', ' predictions', ' mismatches', ' simplifying assumptions', ' complex field settings', ' results', ' salt water', ' wedge', ' coast', ' predevelopment conditions', ' Predictions', ' aquifers', ' salt water', ' wedge', ' inland', ' predevelopment conditions', ' pumping', ' Sharp-interface solutions', ' screen', ' vulnerability', ' coastal', ' aquifers', ' sea water', ' intrusion', ' sources of', ' uncertainty', ' identified', ' study', ' sharp-interface results)\n']}


def getFscore():
  countTrue = 0
  countFalse = 0
  countFalsePredict = 0
  countTruePredict = 0
  for i in true_result[0]:
    if i in test_result3[0] or " "+i in test_result3[0]:
      print("true item",i)
      countTrue+=1
    else:
      print("false item", i)
      countFalse+=1
  for j in test_result3[0]:
    if j not in true_result[0]:
      countFalsePredict +=1
    else:
      countTruePredict+=1
  print(countTrue,countFalse,countFalsePredict,countTruePredict)


if __name__ =="__main__":
  # readfiles()
  # getFscore()
  TP = 71
  FP = 14
  TN = 84
  FN = 1
  P = TP/(TP+FP)
  R = TP/(TP+FN)
  F1 = 2*TP/(2*TP+FP+FN)
  print(P,R,F1)



  # import spacy
  # from scispacy.abbreviation import AbbreviationDetector
  #
  # nlp = spacy.load("en_core_sci_scibert")
  #
  # # Add the abbreviation pipe to the spacy pipeline.
  # nlp.add_pipe("abbreviation_detector")
  #
  # # 27010511
  # doc = nlp(
  #   "Application of an Analytical Solution as a Screening Tool for Sea Water Intrusion. Sea water intrusion into aquifers is problematic in many coastal areas. The physics and chemistry of this issue are complex, and sea water intrusion remains challenging to quantify. Simple assessment tools like analytical models offer advantages of rapid application, but their applicability to field situations is unclear. This study examines the reliability of a popular sharp-interface analytical approach for estimating the extent of sea water in a homogeneous coastal aquifer subjected to pumping and regional flow effects and under steady-state conditions. The analytical model is tested against observations from Canada, the United States, and Australia to assess its utility as an initial approximation of sea water extent for the purposes of rapid groundwater management decision making. The occurrence of sea water intrusion resulting in increased salinity at pumping wells was correctly predicted in approximately 60% of cases. Application of a correction to account for dispersion did not markedly improve the results. Failure of the analytical model to provide correct predictions can be attributed to mismatches between its simplifying assumptions and more complex field settings. The best results occurred where the toe of the salt water wedge is expected to be the closest to the coast under predevelopment conditions. Predictions were the poorest for aquifers where the salt water wedge was expected to extend further inland under predevelopment conditions and was therefore more dispersive prior to pumping. Sharp-interface solutions remain useful tools to screen for the vulnerability of coastal aquifers to sea water intrusion, although the significant sources of uncertainty identified in this study require careful consideration to avoid misinterpreting sharp-interface results.")
  # entity = doc.ents
  # print(entity)
