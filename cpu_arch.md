flowchart LR
    %% 全局样式定义：数据通路用粗实线，控制通路用细虚线
    classDef dataModule fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef ctrlModule fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef mux fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    linkStyle 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 stroke-width:2px,stroke:#01579b;
    linkStyle 100,101,102,103,104,105,106,107,108,109 stroke-width:1px,stroke:#4a148c,stroke-dasharray:5 5;

    %% ======================
    %% 1. 取指级（IF）模块
    %% ======================
    CLK_RST[时钟/复位<br>clk/rst]:::ctrlModule
    PC[PC模块<br>PC]:::dataModule
    ADD_PC4[PC+4加法器<br>ADD]:::dataModule
    IMEM[指令存储器<br>imem]:::dataModule

    CLK_RST --> PC
    PC -->|pc[31:0]| IMEM
    PC -->|pc[31:0]| ADD_PC4
    ADD_PC4 -->|pc4[31:0]| WB_MUX
    ADD_PC4 --> PC_NEXT_MUX
    PC_NEXT_MUX[PC选择MUX<br>MUX2]:::mux
    PC_NEXT_MUX -->|pc_next[31:0]| PC

    %% ======================
    %% 2. 译码级（ID）模块
    %% ======================
    CONTROL[控制单元<br>control_unit]:::ctrlModule
    REGFILE[寄存器堆<br>regfile]:::dataModule
    IMM_GEN[立即数生成<br>imm_gen]:::dataModule

    IMEM -->|instr[31:0]| CONTROL
    IMEM -->|instr[31:0]| IMM_GEN
    IMEM -->|rs1/rs2/rd地址| REGFILE

    CLK_RST --> REGFILE

    %% ======================
    %% 3. 执行级（EX）模块
    %% ======================
    ALU_B_MUX[ALU-B选择MUX<br>MUX2]:::mux
    ALU[算术逻辑单元<br>ALU]:::dataModule

    REGFILE -->|rd1[31:0]| ALU
    REGFILE -->|rd2[31:0]| ALU_B_MUX
    IMM_GEN -->|imm_ext[31:0]| ALU_B_MUX
    ALU_B_MUX -->|alu_b[31:0]| ALU

    %% ======================
    %% 4. 访存级（MEM）模块
    %% ======================
    DMEM[数据存储器<br>dmem]:::dataModule

    ALU -->|alu_result[31:0]| DMEM
    REGFILE -->|rd2[31:0]| DMEM
    CLK_RST --> DMEM

    %% ======================
    %% 5. 写回级（WB）模块
    %% ======================
    WB_MUX[写回选择MUX<br>MUX3]:::mux

    ALU -->|alu_result[31:0]| WB_MUX
    DMEM -->|dmem_rd[31:0]| WB_MUX
    WB_MUX -->|reg_wd[31:0]| REGFILE

    %% ======================
    %% 控制通路（虚线）
    %% ======================
    CONTROL -->|alu_op[3:0]| ALU
    CONTROL -->|imm_src[2:0]| IMM_GEN
    CONTROL -->|alu_src| ALU_B_MUX
    CONTROL -->|mem_we| DMEM
    CONTROL -->|reg_we| REGFILE
    CONTROL -->|wb_sel[1:0]| WB_MUX
    CONTROL -->|branch/jump| PC_NEXT_MUX

    %% 模块分类标注
    note over PC,IMEM: 取指级 IF
    note over CONTROL,IMM_GEN,REGFILE: 译码级 ID
    note over ALU_B_MUX,ALU: 执行级 EX
    note over DMEM: 访存级 MEM
    note over WB_MUX: 写回级 WB