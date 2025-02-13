package require tdom

#get arguments
set result_path [lindex $argv 2]
set result_file [lindex $argv 3]
set result_node_file_path "$result_path/$result_file"

#removes comments from line, source: https://www.rosettacode.org/wiki/Strip_comments_from_a_string#Tcl
proc stripLineComments {inputString {commentChars ";#"}} {
    # Convert the character set into a transformation
    foreach c [split $commentChars ""] {lappend map $c "\uFFFF"}; # *very* rare character!
    # Apply transformation and then use a simpler constant RE to strip
    regsub -all -line {\uFFFF.*$} [string map $map $inputString] "" commentStripped
    # Now strip the whitespace
    regsub -all -line {^[ \t\r]*(.*\S)?[ \t\r]*$} $commentStripped {\1}
}

proc getTreeRoot { fp } {

	set doc [dom parse [read $fp]]
	close $fp

	set root [$doc documentElement]

	return $root
}

proc createTreePath { first_arg args } {

	set tags "/$first_arg"

	foreach arg $args {

		append tags "/$arg"

	}

	return $tags
}

proc getTreeNodesByRelPath { node first_arg args } {

	set tags [createTreePath $first_arg {*}$args]

	set res_nodes [$node selectNodes "[$node toXPath]$tags"]

	return $res_nodes
}

proc getTreeNodesByRelPathInNamespaces { node node_ns first_arg } {

	#TODO add subpath support

	set res_nodes [$node selectNodes -namespaces $node_ns "//[lindex $node_ns 0]:$first_arg"]

	return $res_nodes

}

proc getTreeNodes { root_nodes first_arg args } {

	set res_nodes {}

	foreach node $root_nodes {

		lset res_nodes [getTreeNodesByRelPath $node $first_arg {*}$args]

	}

	return $res_nodes

}

proc getTreeNodesInNamespaces { root_nodes node_ns first_arg } {

	#TODO add subpath support

	set res_nodes {}

	foreach node $root_nodes {

		lset res_nodes [getTreeNodesByRelPathInNamespaces $root_nodes $node_ns $first_arg]

	}

	return $res_nodes
}

proc getTreeNodesByRegularExpression { expression nodes args } {

	set res_nodes {}

	foreach node $nodes {

		set node_data [[getTreeNodesByRelPath $node $args] text]

		if { [regexp $expression $node_data] > 0} {

			lset res_nodes $node

		}

	}

	return $res_nodes
}

proc synthesis { } {

	#first argument is -f, second synth.tcl and third is result path -> base 3

	global argc
	global argv

	#set project path variable
	set project_path [lindex $argv 4]

	puts "Project path: $project_path"

	#set solution name variable
	set solution_name [lindex $argv 5]
	set top_name [lindex $argv 8]

	puts "Project/Solution name: $solution_name"

	#set part and clock period
	set part [lindex $argv 6]
	set clock_period [lindex $argv 7]

	# set if 
	set c_sim [lindex $argv 9]
	set c_synth [lindex $argv 10]
	set rtl_synth [lindex $argv 11]

	#set path of solution report
	set report_path "$solution_name/$solution_name/syn/report"

	#change directory
	cd "$project_path"

	#open project
	open_project $solution_name

	#open solution
	open_solution -flow_target vivado $solution_name

	#set part
	set_part $part

	#set clock period
	create_clock -period $clock_period -name default
	set_clock_uncertainty 12.5%

	#set top function
	set_top $top_name

	#add all synthesis and simulation files
	for {set i 12} {$i < $argc} {incr i} {
		add_files [lindex $argv $i]
		add_files -tb [lindex $argv $i]
	}

	# C Simulation
	if {$c_sim} {
		csim_design -O -clean
	}

	# C Synthesis
	if ($c_synth) {
		csynth_design
		puts "Write report of $solution_name."
		writeSynthResult $report_path
	}

	# RTL Synthesis
	if ($rtl_synth) {
		export_design -flow syn -format ip_catalog -rtl VHDL
	}

	#close solution
	close_solution

	#close project
	close_project


	exit
}

proc writeSynthResult {report_path} {

	#get arguments
	global result_path
	global result_node_file_path
	global result_edge_file_path
	#global node_id
	#global node_name
	#global node_kernel_size
	#global node_vector_size
	global node_loop_amount

	#create path to store results
	file mkdir $result_path

	#open file
	set fp [open "$report_path/csynth.xml" r]

	#set root for synthesis results
	set csynth_report_root [getTreeRoot $fp]

	#open file
	set fp [open "$result_node_file_path" w]

	#get resources
	set resources_node [getTreeNodes $csynth_report_root "AreaEstimates" "Resources"]
	set nr_lut [[getTreeNodesByRelPath $resources_node "LUT"] text]
	set nr_ff [[getTreeNodesByRelPath $resources_node "FF"] text]
	set nr_dsp [[getTreeNodesByRelPath $resources_node "DSP"] text]
	set nr_bram [[getTreeNodesByRelPath $resources_node "BRAM_18K"] text]
	set nr_uram [[getTreeNodesByRelPath $resources_node "URAM"] text]

	#print resources
	#puts -nonewline $fp "$node_id;$node_name;$node_kernel_size;$node_vector_size;$nr_lut;$nr_ff;$nr_dsp;$nr_bram;$nr_uram"
	puts -nonewline $fp "$nr_lut;$nr_ff;$nr_dsp;$nr_bram;$nr_uram"

	# get total latency
	set total_latency [[getTreeNodes $csynth_report_root "PerformanceEstimates" "SummaryOfOverallLatency" "Worst-caseLatency"] text]
	puts -nonewline $fp ";$total_latency"

	# get estimated clock period
	set clock_period [[getTreeNodes $csynth_report_root "PerformanceEstimates" "SummaryOfTimingAnalysis" "EstimatedClockPeriod"] text]
	puts -nonewline $fp ";$clock_period"

	#close file
	puts $fp ""
	close $fp

	puts "Result was successfully written to $result_node_file_path"
}

#main
synthesis

#close shell
exit
