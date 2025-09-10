import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabase = createClient(supabaseUrl, supabaseServiceKey);

// Mock PDF generation - in real implementation, use a PDF library like jsPDF or puppeteer
const generateMockReportUrl = (reportType, sessionCode) => {
  return `https://example.com/reports/${sessionCode}-${reportType}.pdf`;
};

const generateReportContent = (reportType, session, analysis) => {
  const baseContent = {
    session_code: session.session_code,
    patient_name: session.patient?.full_name || 'Unknown Patient',
    doctor_name: session.doctor?.full_name || 'Unknown Doctor',
    analysis_date: analysis.analysis_completed_at,
    prediction: analysis.prediction,
    confidence: (analysis.confidence_score * 100).toFixed(1) + '%'
  };

  switch (reportType) {
    case 'patient':
      return {
        title: 'EEG Analysis Report - Patient Summary',
        summary: `Your EEG analysis has been completed. The automated system has provided an assessment of your brain activity patterns.`,
        recommendation: 'Please discuss these results with your doctor during your next appointment.',
        technical_details: false,
        ...baseContent
      };
    
    case 'doctor':
      return {
        title: 'EEG Analysis Report - Clinical Summary',
        summary: `Comprehensive EEG analysis for ${baseContent.patient_name}. Analysis shows ${analysis.prediction} with ${baseContent.confidence} confidence.`,
        clinical_notes: 'Please review the technical details and correlate with clinical presentation.',
        technical_details: true,
        probabilities: analysis.probabilities,
        stats: analysis.stats_data,
        ...baseContent
      };
    
    case 'technical':
      return {
        title: 'EEG Analysis Report - Technical Details',
        summary: `Full technical analysis report including all computational details and intermediate results.`,
        raw_data: analysis,
        methodology: 'Machine learning analysis using deep neural networks trained on clinical EEG datasets.',
        technical_details: true,
        ...baseContent
      };
    
    default:
      return baseContent;
  }
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { session_id, analysis_result_id } = req.body;

    if (!session_id || !analysis_result_id) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }

    // Get session and analysis details
    const { data: session, error: sessionError } = await supabase
      .from('eeg_sessions')
      .select(`
        *,
        doctor:user_profiles!doctor_id(
          id,
          full_name,
          email,
          doctor_profiles(specialization)
        ),
        patient:user_profiles!patient_id(
          id,
          full_name,
          patient_profiles(patient_id)
        )
      `)
      .eq('id', session_id)
      .single();

    if (sessionError || !session) {
      return res.status(404).json({ error: 'EEG session not found' });
    }

    const { data: analysis, error: analysisError } = await supabase
      .from('eeg_analysis_results')
      .select('*')
      .eq('id', analysis_result_id)
      .single();

    if (analysisError || !analysis) {
      return res.status(404).json({ error: 'Analysis results not found' });
    }

    // Generate reports for all three types
    const reportTypes = ['patient', 'doctor', 'technical'];
    const generatedReports = [];

    for (const reportType of reportTypes) {
      try {
        // Generate report content
        const reportContent = generateReportContent(reportType, session, analysis);
        
        // In a real implementation, generate actual PDF here
        const reportUrl = generateMockReportUrl(reportType, session.session_code);
        
        // Determine who the report is for
        let generatedForUserId;
        let generatedByDoctorId = session.doctor_id;
        
        switch (reportType) {
          case 'patient':
            generatedForUserId = session.patient_id;
            break;
          case 'doctor':
            generatedForUserId = session.doctor_id;
            break;
          case 'technical':
            // Technical reports can be accessed by radiologists and doctors
            generatedForUserId = session.doctor_id; // Default to doctor
            break;
        }

        // Save report record to database
        const reportData = {
          session_id,
          analysis_result_id,
          report_type: reportType,
          report_url: reportUrl,
          generated_for_user_id: generatedForUserId,
          generated_by_doctor_id: generatedByDoctorId,
          is_accessible: true,
          access_expires_at: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000).toISOString(), // 90 days
          generated_at: new Date().toISOString()
        };

        const { data: savedReport, error: reportError } = await supabase
          .from('reports')
          .insert(reportData)
          .select()
          .single();

        if (reportError) {
          console.error(`Error saving ${reportType} report:`, reportError);
          continue;
        }

        generatedReports.push({
          type: reportType,
          id: savedReport.id,
          url: reportUrl,
          generatedFor: generatedForUserId
        });

      } catch (error) {
        console.error(`Error generating ${reportType} report:`, error);
      }
    }

    if (generatedReports.length === 0) {
      return res.status(500).json({ error: 'Failed to generate any reports' });
    }

    // Create notifications for relevant users
    try {
      const notifications = [];
      
      // Notify patient
      notifications.push({
        user_id: session.patient_id,
        title: 'EEG Analysis Complete',
        message: `Your EEG analysis results are now available for review.`,
        type: 'report_ready',
        related_resource_type: 'eeg_session',
        related_resource_id: session_id,
        created_at: new Date().toISOString()
      });

      // Notify doctor
      notifications.push({
        user_id: session.doctor_id,
        title: 'EEG Analysis Complete',
        message: `EEG analysis for ${session.patient?.full_name} has been completed. Clinical and technical reports are available.`,
        type: 'report_ready',
        related_resource_type: 'eeg_session',
        related_resource_id: session_id,
        created_at: new Date().toISOString()
      });

      if (notifications.length > 0) {
        await supabase
          .from('notifications')
          .insert(notifications);
      }
    } catch (error) {
      console.error('Error creating notifications:', error);
    }

    res.status(200).json({
      message: 'Reports generated successfully',
      reports: generatedReports,
      session_id
    });

  } catch (error) {
    console.error('Report generation error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}