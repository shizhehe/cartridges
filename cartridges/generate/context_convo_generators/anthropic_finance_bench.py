# import anthropic

from capsules.generate.context_convo_generators.base import ContextConvoGenerator
from pydrantic import ObjectConfig

from capsules.generate.structs import ContextConvo, Message
from capsules.tasks.finance import FinanceBenchContextConfig

data = [
  {
    "question": "What was AMD's total net revenue for the fiscal year 2022?",
    "answer": "AMD's total net revenue for the fiscal year 2022 was $23.6 billion, which represents a 44% increase compared to 2021 net revenue of $16.4 billion."
  },
  {
    "question": "What major acquisitions did AMD complete in 2022?",
    "answer": "AMD completed two major acquisitions in 2022: Xilinx, Inc. (completed on February 14, 2022) and Pensando Systems, Inc. (completed on May 26, 2022). The Xilinx acquisition expanded AMD's product portfolio to include adaptable hardware platforms, while Pensando added high-performance data processing units (DPUs) and complementary software stack to AMD's offerings."
  },
  {
    "question": "How did AMD's operating income change from 2021 to 2022?",
    "answer": "AMD's operating income decreased from $3.6 billion in 2021 to $1.3 billion in 2022. The decrease was primarily driven by amortization of intangible assets associated with the Xilinx acquisition."
  },
  {
    "question": "What was the value of AMD's cash, cash equivalents and short-term investments at the end of 2022?",
    "answer": "AMD's cash, cash equivalents and short-term investments as of December 31, 2022 were $5.9 billion, compared to $3.6 billion at the end of 2021."
  },
  {
    "question": "What was AMD's total debt as of December 31, 2022?",
    "answer": "AMD's aggregate principal amount of total debt as of December 31, 2022 was $2.5 billion, compared to $313 million as of December 25, 2021."
  },
  {
    "question": "Which segment had the highest operating income for AMD in 2022?",
    "answer": "In 2022, the Embedded segment had the highest operating income for AMD at $2.252 billion, followed by the Data Center segment at $1.848 billion, Client segment at $1.190 billion, and Gaming segment at $953 million."
  },
  {
    "question": "What percentage of AMD's total revenue came from international sales in 2022?",
    "answer": "International sales as a percentage of net revenue were 66% in 2022, down from 72% in 2021."
  },
  {
    "question": "What was the primary reason for the decrease in AMD's Client segment revenue in 2022?",
    "answer": "The primary reason for the 10% decrease in AMD's Client segment revenue in 2022 was lower processor shipments, which decreased by 24%. This was driven by challenging PC market conditions and significant inventory correction across the PC supply chain experienced during the second half of 2022."
  },
  {
    "question": "What was the total amount of stock repurchases by AMD in 2022?",
    "answer": "During 2022, AMD repurchased 36.3 million shares of its common stock under the Repurchase Program for $3.7 billion."
  },
  {
    "question": "What new debt did AMD issue in 2022?",
    "answer": "In June 2022, AMD issued $1.0 billion in aggregate principal amount of senior notes, consisting of $500 million in aggregate principal amount of 3.924% Senior Notes due 2032 and $500 million in aggregate principal amount of 4.393% Senior Notes due 2052."
  },
  {
    "question": "What was AMD's total goodwill as of December 31, 2022?",
    "answer": "AMD's total goodwill as of December 31, 2022 was $24.177 billion, a significant increase from $289 million as of December 25, 2021. This increase was primarily due to acquisitions completed in 2022, particularly Xilinx."
  },
  {
    "question": "What was the purchase consideration for the Xilinx acquisition?",
    "answer": "The total purchase consideration for the Xilinx acquisition was $48.8 billion ($46.4 billion, net of cash acquired of $2.4 billion). The purchase consideration consisted of $48.5 billion of fair value of 429 million shares of AMD's common stock issued to Xilinx stockholders and $275 million of fair value of replacement equity awards attributable to services rendered pre-combination."
  },
  {
    "question": "What was AMD's net income for fiscal year 2022?",
    "answer": "AMD's net income for fiscal year 2022 was $1.3 billion, a decrease from $3.2 billion in 2021. The decrease in net income was primarily driven by lower operating income."
  },
  {
    "question": "What factors drove the growth in AMD's Data Center segment revenue in 2022?",
    "answer": "AMD's Data Center segment revenue increased by 64% in 2022 compared to 2021. The increase was primarily driven by higher sales of AMD's EPYC server processors."
  },
  {
    "question": "How much did AMD spend on marketing and advertising expenses in 2022?",
    "answer": "AMD's total marketing and advertising expenses for 2022 were approximately $683 million, an increase from $578 million in 2021 and $314 million in 2020."
  },
  {
    "question": "What was AMD's tax situation in 2022?",
    "answer": "AMD recorded an income tax benefit of $122 million in 2022, compared to an income tax provision of $513 million in 2021, representing effective tax rates of (10%) and 14%, respectively. The reduction in income tax expense in 2022 was primarily due to the lower pre-tax income coupled with a $261 million foreign-derived intangible income tax benefit and $241 million of research and development tax credits."
  },
  {
    "question": "What was AMD's inventory balance at the end of 2022 and how did it change from 2021?",
    "answer": "AMD's inventory balance was $3.771 billion as of December 31, 2022, compared to $1.955 billion as of December 25, 2021. This represents an increase of approximately $1.816 billion or 93%."
  },
  {
    "question": "What new credit facility did AMD establish in 2022?",
    "answer": "In April 2022, AMD entered into a Revolving Credit Agreement with Wells Fargo Bank, N.A. as administrative agent and other banks as lenders. The agreement provides for a five-year unsecured revolving credit facility in the aggregate principal amount of $3.0 billion. As of December 31, 2022, there were no funds drawn from this facility."
  },
  {
    "question": "What was AMD's gross margin percentage in 2022 and how did it change from 2021?",
    "answer": "AMD's gross margin as a percentage of net revenue was 45% in 2022, compared to 48% in 2021. The decrease in gross margin was primarily due to amortization of intangible assets associated with the Xilinx acquisition."
  },
  {
    "question": "How did AMD's Gaming segment perform in 2022 compared to 2021?",
    "answer": "AMD's Gaming segment net revenue increased by 21% in 2022 to $6.805 billion, compared to $5.607 billion in 2021. The increase was driven by higher semi-custom product sales due to higher demand for gaming console SoCs, partially offset by lower gaming graphics sales. Operating income for the Gaming segment increased slightly to $953 million in 2022 from $934 million in 2021."
  },
  {
    "question": "What were AMD's research and development expenses in 2022 and how did they change from 2021?",
    "answer": "AMD's research and development expenses were $5.005 billion in 2022, an increase of $2.160 billion or 76% compared to $2.845 billion in 2021. The increase was primarily driven by strategic investments across all of AMD's segments, including an increase in headcount through acquisitions and organic growth."
  },
  {
    "question": "What was the value of AMD's acquisition-related intangible assets as of December 31, 2022?",
    "answer": "The value of AMD's acquisition-related intangible assets as of December 31, 2022 was $24.118 billion. This was a significant increase from the previous year, as the balance was not material as of December 25, 2021."
  },
  {
    "question": "How much cash did AMD generate from operating activities in 2022?",
    "answer": "AMD generated $3.565 billion in net cash from operating activities in 2022, slightly higher than the $3.521 billion generated in 2021."
  },
  {
    "question": "What was the size of AMD's workforce at the end of 2022?",
    "answer": "AMD had approximately 25,000 employees in its global workforce as of December 31, 2022. This includes approximately 10,000 new employees who were added to the company through acquisitions and direct hiring during fiscal year 2022."
  },
  {
    "question": "What was the largest component of AMD's current assets as of December 31, 2022?",
    "answer": "The largest component of AMD's current assets as of December 31, 2022 was cash and cash equivalents at $4.835 billion, followed by accounts receivable at $4.126 billion and inventories at $3.771 billion."
  },
  {
    "question": "What were AMD's capital expenditures in 2022?",
    "answer": "AMD's capital expenditures, represented by 'Purchases of property and equipment' in the cash flow statement, were $450 million in 2022, compared to $301 million in 2021."
  },
  {
    "question": "How much did AMD spend on stock-based compensation in 2022?",
    "answer": "AMD recorded $1.081 billion in stock-based compensation expense in 2022, a significant increase from $379 million in 2021."
  },
  {
    "question": "What was the total value of AMD's unconditional purchase commitments as of December 31, 2022?",
    "answer": "AMD's total future unconditional purchase commitments as of December 31, 2022 were $8.610 billion, with $6.489 billion due in 2023."
  },
  {
    "question": "What was AMD's basic earnings per share for 2022?",
    "answer": "AMD's basic earnings per share for 2022 was $0.85, a decrease from $2.61 in 2021."
  },
  {
    "question": "What factors drove the significant increase in AMD's Embedded segment revenue in 2022?",
    "answer": "AMD's Embedded segment revenue increased significantly from $246 million in 2021 to $4.552 billion in 2022. This increase was primarily driven by the inclusion of Xilinx embedded product revenue as a result of the acquisition of Xilinx in February 2022."
  },
  {
    "question": "What was the status of AMD's commercial paper program at the end of 2022?",
    "answer": "In November 2022, AMD established a new commercial paper program where they may issue unsecured commercial paper notes up to a maximum principal amount outstanding at any time of $3.0 billion with a maturity of up to 397 days from the date of issue. As of December 31, 2022, AMD had no commercial paper outstanding."
  },
  {
    "question": "What percentage of AMD's common shares outstanding were repurchased in 2022?",
    "answer": "AMD repurchased 36.3 million shares of common stock in 2022. Based on the beginning of period share count of 1,207 million shares outstanding, this represents approximately 3% of outstanding shares."
  },
  {
    "question": "What was AMD's total stockholders' equity as of December 31, 2022, and how did it change from 2021?",
    "answer": "AMD's total stockholders' equity as of December 31, 2022 was $54.750 billion, a significant increase from $7.497 billion as of December 25, 2021. This increase of $47.253 billion was primarily due to the issuance of common stock for the Xilinx acquisition."
  },
  {
    "question": "What were AMD's marketing, general and administrative expenses in 2022 and how did they change from 2021?",
    "answer": "AMD's marketing, general and administrative expenses were $2.336 billion in 2022, an increase of $888 million or 61% compared to $1.448 billion in 2021. The increase was primarily due to an increase in headcount through acquisitions and organic growth, go-to-market activities, and acquisition-related costs."
  },
  {
    "question": "What was the amortization of acquisition-related intangibles reported by AMD in 2022?",
    "answer": "In 2022, AMD reported a total of $3.548 billion in amortization of acquisition-related intangibles. This consisted of $1.448 billion in cost of sales and $2.100 billion in operating expenses."
  },
  {
    "question": "What new products did AMD introduce in 2022?",
    "answer": "In 2022, AMD introduced several new products including: 4th Gen AMD EPYC processors for data centers; Ryzen 7000 Series Desktop processors powered by the new 'Zen 4' architecture; AMD Ryzen 7000 Mobile processors; Ryzen Threadripper PRO 5000 WX-Series workstation processors; AMD Ryzen PRO 7030 Series Mobile processors; AMD Radeon RX 7900 XTX and Radeon RX 7900 XT gaming graphics cards; and AMD Ryzen Embedded R2000 Series processors for embedded applications."
  },
  {
    "question": "What was AMD's interest expense in 2022 and how did it change from 2021?",
    "answer": "AMD's interest expense was $88 million in 2022, an increase of $54 million compared to $34 million in 2021. The increase was primarily due to interest expense from the 2.95% Senior Notes due 2024 and the 2.375% Senior Notes due 2030 (the Assumed Xilinx Notes) and the 3.924% Notes and 4.393% Notes issued in 2022."
  },
  {
    "question": "What was AMD's diluted earnings per share for 2022?",
    "answer": "AMD's diluted earnings per share for 2022 was $0.84, a decrease from $2.57 in 2021."
  },
  {
    "question": "What was AMD's accumulated deficit as of December 31, 2022, and how did it change from 2021?",
    "answer": "AMD's accumulated deficit was $131 million as of December 31, 2022, a significant decrease from $1.451 billion as of December 25, 2021. This improvement of $1.320 billion equals the company's net income for 2022."
  },
  {
    "question": "How much did AMD spend on depreciation and amortization in 2022?",
    "answer": "AMD spent $4.174 billion on depreciation and amortization in 2022, a significant increase from $407 million in 2021. This increase was primarily due to amortization of intangible assets from the Xilinx acquisition."
  },
  {
    "question": "What were the primary drivers for the changes in AMD's accounts receivable in 2022?",
    "answer": "AMD's accounts receivable increased by $1.420 billion from $2.706 billion in 2021 to $4.126 billion in 2022. According to the cash flow statement, this increase was driven primarily by higher revenue in the fourth quarter of 2022 compared to the fourth quarter of 2021."
  },
  {
    "question": "What was the fair value of AMD's foreign currency forward contracts as of December 31, 2022?",
    "answer": "The fair value of AMD's foreign currency forward contracts as of December 31, 2022 was a net loss of $28 million, compared to a net loss of $2 million as of December 25, 2021."
  },
  {
    "question": "What business segments does AMD report its financial performance under?",
    "answer": "As of 2022, AMD reports its financial performance based on four reportable segments: Data Center segment (which primarily includes server CPUs and GPUs, DPUs, FPGAs and Adaptive SoC products), Client segment (which primarily includes CPUs, APUs, and chipsets for desktop and notebook personal computers), Gaming segment (which primarily includes discrete GPUs, semi-custom SoC products and development services), and Embedded segment (which primarily includes embedded CPUs, GPUs, APUs, FPGAs, and Adaptive SoC products)."
  },
  {
    "question": "What was the purchase price of the Pensando acquisition?",
    "answer": "AMD completed the acquisition of Pensando for a transaction valued at approximately $1.9 billion. The recorded purchase consideration was $1.7 billion, net of deferred cash compensation requiring future services and other customary closing adjustments."
  },
  {
    "question": "How did AMD's accounts payable change from 2021 to 2022?",
    "answer": "AMD's accounts payable increased from $1.321 billion as of December 25, 2021 to $2.493 billion as of December 31, 2022, representing an increase of $1.172 billion or 89%. According to the cash flow statement, the increase was primarily due to timing of payments to suppliers."
  },
  {
    "question": "What was AMD's weighted-average number of shares for basic earnings per share calculation in 2022?",
    "answer": "AMD's weighted-average number of shares for basic earnings per share calculation in 2022 was 1,561 million shares, an increase from 1,213 million shares in 2021."
  },
  {
    "question": "What were the main changes in AMD's Data Center segment operating income from 2021 to 2022?",
    "answer": "AMD's Data Center segment operating income increased from $991 million in 2021 to $1.848 billion in 2022, an increase of $857 million or about 86%. This increase was primarily driven by higher revenue from increased sales of EPYC server processors, partially offset by higher operating expenses."
  },
  {
    "question": "What was AMD's cash position at the end of 2022 compared to the end of 2021?",
    "answer": "AMD's cash and cash equivalents at the end of 2022 were $4.835 billion, compared to $2.535 billion at the end of 2021. This represents an increase of $2.300 billion or about 91%."
  },
  {
    "question": "What was the performance of AMD's Client segment in 2022 compared to 2021?",
    "answer": "AMD's Client segment net revenue decreased by 10% to $6.201 billion in 2022 from $6.887 billion in 2021. The decrease was primarily driven by a 24% decrease in unit shipments due to challenging PC market conditions and inventory corrections, partially offset by a 19% increase in average selling price. Operating income for the Client segment decreased by 43% to $1.190 billion in 2022 from $2.088 billion in 2021."
  }
]

class FinanceBenchClaudeSonnet37Amd(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):
        ...

    def sample_convos(self, start_idx: int, end_idx: int) -> list[ContextConvo]:
        num_convos = end_idx - start_idx
        assert num_convos > 0
        return [
            ContextConvo(
                id=None,
                messages=[
                    Message(role='user', content=data[idx]['question']),
                    Message(role='assistant', content=data[idx]['answer']),
                ],
                type="manual",
                metadata={}
            )
            for idx in range(num_convos)
        ]

