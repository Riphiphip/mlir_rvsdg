
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

// Create a mem state for testing memory effect interface.
%ptr, %root_mem_state = jlm.alloca i32 () -> !llvm.ptr<i32>, !rvsdg.memState

%_2 = arith.constant 10: i32
%_3 = arith.constant 10: i32

// Identical theta nodes
%_theta_output, %_theta_output2, %memState = rvsdg.thetaNode (%_2: i32, %_3: i32, %root_mem_state: !rvsdg.memState): 
    (%_2: i32, %_3: i32, %root_mem_state: !rvsdg.memState): {
        %_predicate = rvsdg.constantCtrl 0:<2>
        rvsdg.thetaResult (%_predicate): (%_3:i32, %_2:i32, %root_mem_state: !rvsdg.memState)
    }->i32, i32, !rvsdg.memState

%_theta_output3, %_theta_output4, %memState2 = rvsdg.thetaNode (%_2: i32, %_3: i32, %root_mem_state: !rvsdg.memState): 
    (%_2: i32, %_3: i32, %root_mem_state: !rvsdg.memState): {
        %_predicate = rvsdg.constantCtrl 0:<2>
        rvsdg.thetaResult (%_predicate): (%_3:i32, %_2:i32, %root_mem_state: !rvsdg.memState)
    }->i32, i32, !rvsdg.memState

%_4 = arith.addi %_theta_output, %_theta_output2: i32
%_5 = arith.addi %_theta_output3, %_theta_output4: i32

%_constant_predicate = rvsdg.constantCtrl 0:<2>
%_gamma_output:2 = rvsdg.gammaNode(%_constant_predicate:<2>) (%_4:i32, %_5:i32):[
    (%_arg0: i32, %_arg1: i32):{
        rvsdg.gammaResult(%_arg0:i32, %_arg1:i32)
    },
    (%_arg0: i32, %_arg1: i32):{
        rvsdg.gammaResult(%_arg0:i32, %_arg1:i32)
    }
] -> i32, i32