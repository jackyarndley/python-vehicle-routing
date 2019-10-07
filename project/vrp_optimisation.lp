\* VRP Optimisation *\
Minimize
OBJ: 375 route_0 + 375 route_1 + 480 route_10 + 345 route_11 + 240 route_12
 + 345 route_13 + 360 route_14 + 345 route_15 + 360 route_16 + 345 route_17
 + 255 route_18 + 420 route_19 + 375 route_2 + 345 route_20 + 345 route_21
 + 315 route_22 + 315 route_23 + 330 route_24 + 330 route_25 + 315 route_26
 + 285 route_27 + 330 route_28 + 375 route_29 + 450 route_3 + 270 route_30
 + 315 route_31 + 390 route_32 + 345 route_33 + 465 route_34 + 330 route_35
 + 375 route_36 + 345 route_37 + 315 route_38 + 300 route_39 + 390 route_4
 + 330 route_40 + 315 route_41 + 345 route_42 + 450 route_43 + 360 route_44
 + 285 route_45 + 1200 route_46 + 1200 route_47 + 1200 route_48
 + 1200 route_49 + 360 route_5 + 1200 route_50 + 1200 route_51 + 1200 route_52
 + 1200 route_53 + 1200 route_54 + 1200 route_55 + 1200 route_56
 + 1200 route_57 + 1200 route_58 + 1200 route_59 + 240 route_6 + 1200 route_60
 + 1200 route_61 + 1200 route_62 + 1200 route_63 + 1200 route_64
 + 1200 route_65 + 1200 route_66 + 1200 route_67 + 1200 route_68
 + 1200 route_69 + 375 route_7 + 1200 route_70 + 1200 route_71 + 1200 route_72
 + 1200 route_73 + 1200 route_74 + 1200 route_75 + 1200 route_76
 + 1200 route_77 + 1200 route_78 + 1200 route_79 + 465 route_8 + 1200 route_80
 + 1200 route_81 + 1200 route_82 + 1200 route_83 + 1200 route_84
 + 1200 route_85 + 1200 route_86 + 1200 route_87 + 1200 route_88
 + 1200 route_89 + 255 route_9 + 1200 route_90 + 1200 route_91
Subject To
_C1: route_0 + route_46 = 1
_C10: route_18 + route_55 + route_64 + route_9 = 1
_C11: route_10 + route_4 + route_50 + route_56 = 1
_C12: route_11 + route_57 = 1
_C13: route_12 + route_52 + route_58 + route_6 = 1
_C14: route_13 + route_59 = 1
_C15: route_14 + route_60 = 1
_C16: route_15 + route_61 = 1
_C17: route_13 + route_16 + route_59 + route_62 = 1
_C18: route_17 + route_37 + route_63 + route_83 = 1
_C19: route_18 + route_55 + route_64 + route_9 = 1
_C2: route_1 + route_43 + route_47 + route_89 = 1
_C20: route_19 + route_65 = 1
_C21: route_20 + route_44 + route_66 + route_90 = 1
_C22: route_21 + route_67 = 1
_C23: route_22 + route_41 + route_68 + route_87 = 1
_C24: route_23 + route_69 = 1
_C25: route_24 + route_39 + route_70 + route_85 = 1
_C26: route_25 + route_71 = 1
_C27: route_26 + route_38 + route_72 + route_84 = 1
_C28: route_27 + route_45 + route_73 + route_91 = 1
_C29: route_28 + route_35 + route_74 + route_81 = 1
_C3: route_2 + route_48 = 1
_C30: route_29 + route_75 = 1
_C31: route_30 + route_76 = 1
_C32: route_31 + route_77 = 1
_C33: route_32 + route_78 = 1
_C34: route_33 + route_42 + route_79 + route_88 = 1
_C35: route_10 + route_19 + route_3 + route_34 + route_43 + route_49
 + route_54 + route_56 + route_65 + route_8 + route_80 + route_89 = 1
_C36: route_14 + route_16 + route_2 + route_20 + route_21 + route_26
 + route_28 + route_29 + route_35 + route_36 + route_38 + route_48 + route_53
 + route_60 + route_62 + route_66 + route_67 + route_7 + route_72 + route_74
 + route_75 + route_81 + route_82 + route_84 = 1
_C37: route_2 + route_36 + route_44 + route_48 + route_53 + route_7 + route_82
 + route_90 = 1
_C38: route_11 + route_14 + route_15 + route_16 + route_17 + route_22
 + route_25 + route_30 + route_31 + route_37 + route_41 + route_45 + route_5
 + route_51 + route_57 + route_60 + route_61 + route_62 + route_63 + route_68
 + route_71 + route_76 + route_77 + route_83 + route_87 + route_91 = 1
_C39: route_14 + route_16 + route_21 + route_25 + route_26 + route_28
 + route_29 + route_35 + route_38 + route_60 + route_62 + route_67 + route_71
 + route_72 + route_74 + route_75 + route_81 + route_84 = 1
_C4: route_0 + route_3 + route_46 + route_49 = 1
_C40: route_23 + route_24 + route_33 + route_39 + route_40 + route_42
 + route_69 + route_70 + route_79 + route_85 + route_86 + route_88 = 1
_C41: route_11 + route_23 + route_27 + route_39 + route_40 + route_57
 + route_69 + route_73 + route_85 + route_86 = 1
_C42: route_15 + route_17 + route_22 + route_37 + route_41 + route_5
 + route_51 + route_61 + route_63 + route_68 + route_83 + route_87 = 1
_C43: route_1 + route_10 + route_19 + route_24 + route_3 + route_32 + route_33
 + route_34 + route_42 + route_43 + route_47 + route_49 + route_54 + route_56
 + route_65 + route_70 + route_78 + route_79 + route_8 + route_80 + route_88
 + route_89 = 1
_C44: route_1 + route_10 + route_3 + route_32 + route_34 + route_43 + route_47
 + route_49 + route_54 + route_56 + route_78 + route_8 + route_80 + route_89
 = 1
_C45: route_17 + route_2 + route_20 + route_31 + route_36 + route_37
 + route_44 + route_48 + route_5 + route_51 + route_53 + route_63 + route_66
 + route_7 + route_77 + route_82 + route_83 + route_90 = 1
_C46: route_1 + route_11 + route_15 + route_27 + route_30 + route_40
 + route_45 + route_47 + route_57 + route_61 + route_73 + route_76 + route_86
 + route_91 = 1
_C47: route_0 + route_1 + route_10 + route_11 + route_12 + route_13 + route_14
 + route_15 + route_16 + route_17 + route_18 + route_19 + route_2 + route_20
 + route_21 + route_22 + route_23 + route_24 + route_25 + route_26 + route_27
 + route_28 + route_29 + route_3 + route_30 + route_31 + route_32 + route_33
 + route_34 + route_35 + route_36 + route_37 + route_38 + route_39 + route_4
 + route_40 + route_41 + route_42 + route_43 + route_44 + route_45 + route_5
 + route_6 + route_7 + route_8 + route_9 <= 20
_C5: route_4 + route_50 = 1
_C6: route_5 + route_51 = 1
_C7: route_12 + route_40 + route_52 + route_58 + route_6 + route_86 = 1
_C8: route_36 + route_53 + route_7 + route_82 = 1
_C9: route_34 + route_54 + route_8 + route_80 = 1
Binaries
route_0
route_1
route_10
route_11
route_12
route_13
route_14
route_15
route_16
route_17
route_18
route_19
route_2
route_20
route_21
route_22
route_23
route_24
route_25
route_26
route_27
route_28
route_29
route_3
route_30
route_31
route_32
route_33
route_34
route_35
route_36
route_37
route_38
route_39
route_4
route_40
route_41
route_42
route_43
route_44
route_45
route_46
route_47
route_48
route_49
route_5
route_50
route_51
route_52
route_53
route_54
route_55
route_56
route_57
route_58
route_59
route_6
route_60
route_61
route_62
route_63
route_64
route_65
route_66
route_67
route_68
route_69
route_7
route_70
route_71
route_72
route_73
route_74
route_75
route_76
route_77
route_78
route_79
route_8
route_80
route_81
route_82
route_83
route_84
route_85
route_86
route_87
route_88
route_89
route_9
route_90
route_91
End
