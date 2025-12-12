import React from 'react';
import { Box, Typography, Button, Grid, Paper, Stack } from '@mui/material';
import { motion } from 'framer-motion';
import ScienceIcon from '@mui/icons-material/Science';
import SpeedIcon from '@mui/icons-material/Speed';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import PsychologyIcon from '@mui/icons-material/Psychology';
import TuneIcon from '@mui/icons-material/Tune';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DescriptionIcon from '@mui/icons-material/Description';
import { Link as ScrollLink } from 'react-scroll';

const FeatureCard = ({ icon, title, description, delay }) => (
    <Paper
        component={motion.div}
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay }}
        sx={{
            p: 4,
            height: '100%',
            width: '100%', // Force box to fill Grid Column
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            justifyContent: 'flex-start',
            textAlign: 'left',
            borderRadius: 4,
            background: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(0,0,0,0.05)',
            boxShadow: '0 4px 6px rgba(0,0,0,0.02)',
            transition: 'all 0.3s ease',
            minHeight: '320px', // Enforce large enough height for uniformity
            '&:hover': {
                transform: 'translateY(-10px)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
            }
        }}
    >
        <Box sx={{ color: 'primary.main', mb: 3, mt: 0, p: 1.5, bgcolor: 'rgba(2,136,209,0.1)', borderRadius: '50%' }}>
            {icon}
        </Box>
        <Typography variant="h6" gutterBottom fontWeight="bold" sx={{ mb: 1, minHeight: '60px', display: 'flex', alignItems: 'center', lineHeight: 1.2 }}>
            {title}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ width: '100%', overflowWrap: 'break-word' }}>
            {description}
        </Typography>
    </Paper>
);

const Home = () => {
    return (
        <Box sx={{ position: 'relative', overflow: 'hidden', pt: 15, pb: 10 }}>
            {/* Background Elements */}
            <Box
                component={motion.div}
                animate={{
                    scale: [1, 1.2, 1],
                    rotate: [0, 10, -10, 0],
                }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                sx={{
                    position: 'absolute',
                    top: '-20%',
                    right: '-10%',
                    width: '600px',
                    height: '600px',
                    borderRadius: '50%',
                    background: 'radial-gradient(circle, rgba(2,136,209,0.1) 0%, rgba(255,255,255,0) 70%)',
                    zIndex: -1
                }}
            />

            <Grid container spacing={6} alignItems="center">
                <Grid item xs={12} md={5}>
                    <motion.div
                        initial={{ opacity: 0, x: -50 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <Typography variant="overline" color="secondary" fontWeight="bold" letterSpacing={2}>
                            Next-Gen Materials Science
                        </Typography>
                        <Typography variant="h1" gutterBottom sx={{ background: 'linear-gradient(45deg, #0288d1 30%, #e91e63 90%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', mb: 2, fontSize: { xs: '2.5rem', md: '3.5rem' } }}>
                            Discover Lead-Free Piezoelectrics
                        </Typography>
                        <Typography variant="h5" color="text.secondary" paragraph sx={{ mb: 4, lineHeight: 1.6 }}>
                            Accelerate your research with AI-driven property prediction. Instantly analyze d<sub>33</sub> and T<sub>c</sub> for complex chemical compositions.
                        </Typography>

                        <Stack direction="row" spacing={2}>
                            <Button
                                component={ScrollLink}
                                to="predict"
                                smooth={true}
                                duration={500}
                                variant="contained"
                                size="large"
                                sx={{ px: 4, py: 1.5, fontSize: '1.1rem' }}
                            >
                                Start Predicting
                            </Button>
                            <Button
                                component={ScrollLink}
                                to="insights"
                                smooth={true}
                                duration={500}
                                variant="outlined"
                                size="large"
                                sx={{ px: 4, py: 1.5, fontSize: '1.1rem' }}
                            >
                                View Insights
                            </Button>
                        </Stack>
                    </motion.div>
                </Grid>

                <Grid item xs={12} md={7}>
                    <Grid container spacing={3} alignItems="stretch">
                        {[
                            { title: "Virtual Lab", icon: <ScienceIcon sx={{ fontSize: 40 }} />, description: "Test thousands of compositions without physical synthesis.", delay: 0.2 },
                            { title: "Instant Results", icon: <SpeedIcon sx={{ fontSize: 40 }} />, description: "Get property predictions in milliseconds using advanced ML models.", delay: 0.3 },
                            { title: "Data Insights", icon: <AutoGraphIcon sx={{ fontSize: 40 }} />, description: "Visualize trends and model performance with interactive charts.", delay: 0.4 },
                            { title: "Custom Models", icon: <TuneIcon sx={{ fontSize: 40 }} />, description: "Fine-tune algorithms with your own parameters for specific needs.", delay: 0.5 },
                            { title: "Model Recommendation System", icon: <PsychologyIcon sx={{ fontSize: 40 }} />, description: "Intelligently selects the optimal model for superior prediction accuracy.", delay: 0.6 },
                            { title: "Export Reports", icon: <DescriptionIcon sx={{ fontSize: 40 }} />, description: "Generate comprehensive PDF reports of your findings instantly.", delay: 0.7 }
                        ].map((feature, index) => (
                            <Grid size={{ xs: 12, sm: 6, md: 4 }} key={index} sx={{ display: 'flex' }}>
                                <FeatureCard
                                    icon={feature.icon}
                                    title={feature.title}
                                    description={feature.description}
                                    delay={feature.delay}
                                />
                            </Grid>
                        ))}
                    </Grid>
                </Grid>
            </Grid>
        </Box>
    );
};

export default Home;
