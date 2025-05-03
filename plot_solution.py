import graphviz
import os
from typing import List # Added for type hinting

# Ensure the output directory exists
output_dir = "diagram_output"
os.makedirs(output_dir, exist_ok=True)
# Define base filename (without extension)
output_filename_base = os.path.join(output_dir, "prompt_evolution_detailed")

# --- Create Graphviz Digraph ---
# Use 'TB' for top-to-bottom ranking. Adjust node/edge separation.
dot = graphviz.Digraph('PromptEvolutionDetailed', comment='NSGA-II for Prompt Optimization', format='png')
dot.attr(rankdir='TB', nodesep='0.3', ranksep='0.2', splines='ortho') # Ortho splines, adjust spacing
dot.attr(labelloc='t', label=' Evolutionary search engine using NSGA-II', fontsize='18')
dot.attr('node', shape='box', style='rounded,filled', fillcolor='azure2', fontname='Helvetica', fontsize='10')
dot.attr('edge', fontname='Helvetica', fontsize='9')

# --- Define Main Workflow Nodes ---
dot.node('init', 'Initialize Population P0\n(Prompt and model \nsampling)')
dot.node('parents', 'Parent Population Pt')
dot.node('operators', 'Genetic Operators') # Label probabilities on edges leading out
dot.node('offspring', 'Offspring Population Qt')
dot.node('combine', 'Combine Populations\nRt = Pt U Qt')
dot.node('evaluate', 'Evaluate Individuals\nMinimize F(p, m)') # General objective
dot.node('selection', 'NSGA-II Selection\n(Rank & Crowding)')
dot.node('next_gen', 'Next Parent Population Pt+1')
dot.node('final_front', 'Final Non-dominated Set\n(Pareto Front)')

# --- Define Representative Individual (More Detailed) ---
# Use HTML-like labels for better structure within the node

# --- MODIFICATION START ---
# List component names as plain text, avoid using angle brackets inside the HTML label
content_components: List[str] = ['context', 'req', 'instr', 'examples', 'cot']
attr_list_str = ", ".join(content_components) # Simple comma-separated string

individual_label = f'''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" ALIGN="LEFT">
  <TR><TD BGCOLOR="lightblue" COLSPAN="2"><B>Prompt Individual (x)</B></TD></TR>
  <TR><TD ALIGN="LEFT">Model (m):</TD><TD ALIGN="LEFT">[Model ID]</TD></TR>
  <TR><TD ALIGN="LEFT">Prompt (p):</TD><TD ALIGN="LEFT">Grammar Rule + Content</TD></TR>
  <TR><TD ALIGN="LEFT" COLSPAN="2" BALIGN="LEFT">  - Content Components: [{attr_list_str}]</TD></TR>
</TABLE>>'''
# --- MODIFICATION END ---

dot.node('ind_example', individual_label, shape='none', margin='0')
# Optional: Place it somewhere visible, maybe connect with dotted line from 'parents'
# dot.edge('parents', 'ind_example', style='dotted', arrowhead='none', constraint='false')


# --- Define Edges (Workflow) with Operator Probabilities ---
dot.edge('init', 'parents', label='t=0\n\n\n')
dot.edge('parents', 'operators')

# Edge from Operators to Offspring showing probabilities
# Use placeholders for probabilities - replace with actual variables/values if generating dynamically
# Example placeholder values used here
prob_cx = 0.9
prob_mut = 0.1
prob_cx_model = 0.5
prob_cx_attr_swap = 0.5
prob_mut_model = 0.1
prob_mut_param = 0.2

edge_label = (
    f'\n\n\n\n\n\n\n\n\nCrossover (prob_cx_model,prob_cx_attr_swap)\n'
    f'Mutation (prob_mut_model, prob_mut_attr)\n'
    
    
)
dot.edge('operators', 'offspring', label=edge_label, fontsize='8')


# Combine populations
with dot.subgraph() as s:
    s.attr(rank='same')
    s.edge('parents', 'combine', arrowhead='none')
    s.edge('offspring', 'combine')
dot.edge('combine', 'evaluate')
dot.edge('evaluate', 'selection')
dot.edge('selection', 'next_gen')
dot.edge('next_gen', 'parents', label=' t = t + 1') # Loop back

# Final Output
dot.edge('selection', 'final_front', label='Termination')


# --- Render and Save ---
try:
    png_filename = output_filename_base + ".png"
    # Specify format='png' explicitly in render
    dot.render(output_filename_base, view=False, engine='dot', format='png', cleanup=True)
    print(f"Diagram saved as {png_filename}")
except Exception as e:
     print(f"Error rendering PNG diagram. Is Graphviz installed and in PATH? Error: {e}")

# --- Save DOT source ---
dot_filename = output_filename_base + '.gv'
try:
    dot.save(dot_filename)
    print(f"DOT source saved as {dot_filename}")
    print(f"To convert DOT to EPS (if Graphviz CLI is installed): dot -Teps {dot_filename} -o {output_filename_base}.eps")
except Exception as save_e:
    print(f"Error saving DOT source: {save_e}")
