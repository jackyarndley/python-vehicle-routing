\* VRP Testing *\
Minimize
OBJ: Route_0 + Route_1 + Route_10 + Route_11 + Route_12 + Route_13 + Route_14
 + Route_15 + Route_16 + Route_17 + Route_18 + Route_19 + Route_2 + Route_20
 + Route_21 + Route_22 + Route_23 + Route_24 + Route_25 + Route_26 + Route_27
 + Route_28 + Route_29 + Route_3 + Route_30 + Route_31 + Route_32 + Route_33
 + Route_34 + Route_35 + Route_36 + Route_37 + Route_38 + Route_39 + Route_4
 + Route_40 + Route_41 + Route_42 + Route_43 + Route_44 + Route_45 + Route_5
 + Route_6 + Route_7 + Route_8 + Route_9
Subject To
_C1: Route_0 + Route_19 + Route_3 + Route_32 + Route_34 + Route_42 + Route_43
 + Route_8 >= 1
_C10: Route_1 + Route_15 + Route_18 + Route_4 + Route_45 + Route_5 + Route_9
 >= 1
_C11: Route_0 + Route_1 + Route_10 + Route_19 + Route_3 + Route_32 + Route_34
 + Route_4 + Route_43 + Route_8 >= 1
_C12: Route_11 + Route_12 + Route_25 + Route_27 + Route_30 + Route_40
 + Route_45 + Route_6 >= 1
_C13: Route_11 + Route_12 + Route_23 + Route_27 + Route_40 + Route_45
 + Route_6 >= 1
_C14: Route_13 + Route_16 + Route_29 >= 1
_C15: Route_13 + Route_14 + Route_21 + Route_25 + Route_26 + Route_28
 + Route_29 + Route_35 + Route_38 >= 1
_C16: Route_15 + Route_17 + Route_18 + Route_30 + Route_31 + Route_37
 + Route_5 + Route_9 >= 1
_C17: Route_13 + Route_14 + Route_16 + Route_21 + Route_26 + Route_29
 + Route_38 >= 1
_C18: Route_15 + Route_17 + Route_22 + Route_31 + Route_37 + Route_41
 + Route_5 >= 1
_C19: Route_1 + Route_10 + Route_18 + Route_32 + Route_4 + Route_43 + Route_45
 + Route_9 >= 1
_C2: Route_1 + Route_10 + Route_18 + Route_32 + Route_4 + Route_43 >= 1
_C20: Route_0 + Route_1 + Route_10 + Route_19 + Route_3 + Route_32 + Route_33
 + Route_34 + Route_4 + Route_42 + Route_43 + Route_8 >= 1
_C21: Route_2 + Route_20 + Route_28 + Route_35 + Route_36 + Route_44 + Route_7
 >= 1
_C22: Route_13 + Route_16 + Route_21 + Route_25 + Route_29 + Route_38 >= 1
_C23: Route_17 + Route_22 + Route_31 + Route_37 + Route_41 + Route_5 >= 1
_C24: Route_12 + Route_23 + Route_24 + Route_33 + Route_39 + Route_40
 + Route_6 >= 1
_C25: Route_23 + Route_24 + Route_33 + Route_39 + Route_40 + Route_42 >= 1
_C26: Route_11 + Route_14 + Route_21 + Route_25 + Route_27 + Route_30 >= 1
_C27: Route_13 + Route_14 + Route_16 + Route_21 + Route_25 + Route_26
 + Route_28 + Route_29 + Route_35 + Route_38 >= 1
_C28: Route_11 + Route_12 + Route_18 + Route_23 + Route_24 + Route_27
 + Route_33 + Route_39 + Route_45 + Route_9 >= 1
_C29: Route_2 + Route_20 + Route_26 + Route_28 + Route_35 + Route_36
 + Route_38 + Route_7 >= 1
_C3: Route_2 + Route_20 + Route_26 + Route_28 + Route_35 + Route_36 + Route_44
 + Route_7 >= 1
_C30: Route_13 + Route_16 + Route_21 + Route_29 >= 1
_C31: Route_11 + Route_15 + Route_25 + Route_27 + Route_30 + Route_37
 + Route_45 + Route_6 >= 1
_C32: Route_14 + Route_15 + Route_17 + Route_22 + Route_30 + Route_31
 + Route_37 + Route_41 + Route_44 >= 1
_C33: Route_0 + Route_1 + Route_10 + Route_19 + Route_3 + Route_32 + Route_34
 + Route_4 + Route_42 + Route_43 + Route_8 >= 1
_C34: Route_23 + Route_24 + Route_33 + Route_39 + Route_42 >= 1
_C35: Route_0 + Route_19 + Route_3 + Route_34 + Route_8 >= 1
_C36: Route_14 + Route_16 + Route_2 + Route_20 + Route_26 + Route_28
 + Route_35 + Route_36 + Route_38 + Route_44 + Route_7 >= 1
_C37: Route_2 + Route_20 + Route_36 + Route_44 + Route_7 >= 1
_C38: Route_15 + Route_17 + Route_18 + Route_22 + Route_30 + Route_31
 + Route_37 + Route_41 + Route_5 + Route_9 >= 1
_C39: Route_13 + Route_14 + Route_16 + Route_21 + Route_25 + Route_26
 + Route_28 + Route_29 + Route_35 + Route_38 >= 1
_C4: Route_0 + Route_10 + Route_19 + Route_3 + Route_34 + Route_8 >= 1
_C40: Route_23 + Route_24 + Route_33 + Route_39 + Route_40 + Route_42 >= 1
_C41: Route_12 + Route_23 + Route_24 + Route_39 + Route_40 + Route_6 >= 1
_C42: Route_17 + Route_22 + Route_31 + Route_37 + Route_41 + Route_5 >= 1
_C43: Route_24 + Route_33 + Route_39 + Route_42 >= 1
_C44: Route_1 + Route_10 + Route_32 + Route_43 >= 1
_C45: Route_2 + Route_20 + Route_22 + Route_31 + Route_36 + Route_41
 + Route_44 + Route_7 >= 1
_C46: Route_11 + Route_12 + Route_18 + Route_27 + Route_30 + Route_45
 + Route_6 + Route_9 >= 1
_C5: Route_4 >= 1
_C6: Route_15 + Route_17 + Route_22 + Route_41 + Route_5 + Route_9 >= 1
_C7: Route_11 + Route_12 + Route_27 + Route_40 + Route_6 >= 1
_C8: Route_2 + Route_20 + Route_36 + Route_44 + Route_7 >= 1
_C9: Route_0 + Route_19 + Route_3 + Route_34 + Route_8 >= 1
Binaries
Route_0
Route_1
Route_10
Route_11
Route_12
Route_13
Route_14
Route_15
Route_16
Route_17
Route_18
Route_19
Route_2
Route_20
Route_21
Route_22
Route_23
Route_24
Route_25
Route_26
Route_27
Route_28
Route_29
Route_3
Route_30
Route_31
Route_32
Route_33
Route_34
Route_35
Route_36
Route_37
Route_38
Route_39
Route_4
Route_40
Route_41
Route_42
Route_43
Route_44
Route_45
Route_5
Route_6
Route_7
Route_8
Route_9
End
