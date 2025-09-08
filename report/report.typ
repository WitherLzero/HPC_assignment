#set document(
  title: "CITS3402/CITS5507 Assignment 1 Report",
  author: "Your Name",
  date: datetime.today(),
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2cm),
  numbering: "1",
)

#set text(
  font: "Linux Libertine",  // Professional serif font, built-in to Typst
  size: 11pt,
  lang: "en",
)

#set heading(numbering: "1.1.")

#set par(justify: true, leading: 0.6em)

// Title page
#align(center + top)[
  #v(15%)  // Position at 1/4 of the page from top
  // UWA Logo
  #image("uwa-logo.png", width: 60%)
]

#align(center)[
  #v(10%)  // Adjust this value to move content up or down
  
  #text(size: 20pt, weight: "bold")[
    CITS3402/CITS5507 Assignment 1
  ]
  
  #text(size: 16pt, style: "italic")[
    Semester 2, 2025
  ]


  #v(2em)
  
  #text(size: 16pt, weight: "bold")[
    Fast Parallel 2D Discrete Convolution Implementation
  ]
  

  
  #v(6em)
  
  #rect(
    width: 65%,
    stroke: (thickness: 1pt, dash: "dashed"),
    inset: 1.2em,
  )[
    #text(size: 14pt, weight: "bold")[
      *Submitted in group:*
    ]
    
    #v(0.5em)
    
    #text(size: 12pt)[
      Your Name, Your Student Number \
      Teammate Name, Teammate Student Number
    ]
  ]


]

#pagebreak()

// Table of contents
#outline()

#pagebreak()


= Introduction

== Problem Statement and Objectives

2D convolution is a fundamental mathematical operation extensively used in signal processing, computer vision, and machine learning applications. In convolutional neural networks (CNNs), hundreds of thousands of convolution operations are performed during inference on high-resolution images, making computational efficiency critical for practical applications.

The discrete 2D convolution of an input feature map $f$ and kernel $g$ is mathematically defined as:

$ (f * g)[n,m] = sum_(i=-M)^M sum_(j=-N)^N f[n+i, m+j] dot g[i,j] $

#figure(
  image("figures/convolution_example.png", width: 80%),
  caption: [Example of 2D convolution operation ]
) <fig:convolution-example>




This assignment focuses on developing a high-performance parallel implementation of 2D convolution with "same" padding using OpenMP. The primary objectives are:

- Implement both serial and parallel versions of 2D convolution
- Achieve significant speedup through effective parallelization
- Analyze performance characteristics and scalability
- Evaluate memory layout and cache optimization strategies

== Testing Environment

Performance analysis was conducted using a two-stage approach to ensure comprehensive evaluation and result validation.

*Primary Testing Environment - Kaya HPC Cluster:*
All primary performance analysis and scalability testing were conducted on the Kaya high-performance computing cluster, which provides:
- Multi-core Intel processors with consistent performance characteristics
- Hierarchical memory system with multiple cache levels (L1, L2, L3)
- OpenMP-enabled GCC compiler environment
- Controlled computational resources for reliable benchmarking
- Support for multi-threading analysis up to 16+ cores

*Development and Validation Environment:*
Initial development, debugging, and correctness verification were performed on local development machines to enable rapid iteration and testing. Local testing ensured code correctness across different system configurations before deployment to Kaya for performance analysis.



== Report Overview



