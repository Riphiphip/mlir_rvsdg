
%2 = arith.constant 10: i32
%3 = arith.constant 10: i32

// Identical theta nodes
%theta_output, %theta_output2 = rvsdg.thetaNode (%2: i32, %3: i32): 
    (%2: i32, %3: i32): {
        %predicate = rvsdg.constantCtrl 0:<2>
        rvsdg.thetaResult (%predicate): (%3:i32, %2:i32)
    }->i32, i32

%theta_output3, %theta_output4 = rvsdg.thetaNode (%2: i32, %3: i32): 
    (%2: i32, %3: i32): {
        %predicate = rvsdg.constantCtrl 0:<2>
        rvsdg.thetaResult (%predicate): (%3:i32, %2:i32)
    }->i32, i32



%4 = arith.addi %theta_output, %theta_output2: i32
%5 = arith.addi %theta_output3, %theta_output4: i32

%constant_predicate = rvsdg.constantCtrl 0:<2>
%gamma_output:2 = rvsdg.gammaNode(%constant_predicate:<2>) (%4:i32, %5:i32):[
    (%arg0: i32, %arg1: i32):{
        rvsdg.gammaResult(%arg0:i32, %arg1:i32)
    },
    (%arg0: i32, %arg1: i32):{
        rvsdg.gammaResult(%arg0:i32, %arg1:i32)
    }
] -> i32, i32