import React from 'react';
import { Box, Container } from '@mui/material';
import { motion } from 'framer-motion';
import Layout from './components/Layout';
import Home from './components/Home';
import Prediction from './components/Prediction';
import ModelInsights from './components/ModelInsights';
import Retraining from './components/Retraining';
import DatasetViewer from './components/DatasetViewer';
import ProjectInfo from './components/ProjectInfo';

const Section = ({ id, children, bgColor = 'transparent' }) => {
  return (
    <Box
      id={id}
      component={motion.div}
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.8 }}
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        py: 8,
        bgcolor: bgColor,
      }}
    >
      <Container maxWidth="xl">
        {children}
      </Container>
    </Box>
  );
};

function App() {
  const [refreshInsights, setRefreshInsights] = React.useState(0);

  const handleTrainingComplete = () => {
    setRefreshInsights(prev => prev + 1);
  };

  return (
    <Layout>
      <Section id="home">
        <Home />
      </Section>

      <Section id="predict" bgColor="#f0f4f8">
        <Prediction />
      </Section>

      <Section id="insights">
        <ModelInsights refreshTrigger={refreshInsights} />
      </Section>

      <Section id="train" bgColor="#f0f4f8">
        <Retraining onTrainingComplete={handleTrainingComplete} />
      </Section>

      <Section id="dataset">
        <DatasetViewer />
      </Section>

      <Section id="about" bgColor="#f0f4f8">
        <ProjectInfo />
      </Section>
    </Layout>
  );
}

export default App;
