import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { Upload, TrendingUp, BarChart3, Calendar, Users, Code, Briefcase } from 'lucide-react';
import * as Papa from 'papaparse';
import _ from 'lodash';

const KaggleSurveyAnalyzer = () => {
  const [data, setData] = useState({
    multipleChoice: null,
    schema: null,
    freeForm: null
  });
  const [analysis, setAnalysis] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(false);

  // Color palette for charts
  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0', '#ffb347'];

  const loadFile = async (fileName, dataType) => {
    try {
      const fileData = await window.fs.readFile(fileName, { encoding: 'utf8' });
      const parsed = Papa.parse(fileData, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: true,
        delimitersToGuess: [',', '\t', '|', ';']
      });
      
      setData(prev => ({
        ...prev,
        [dataType]: parsed.data
      }));
      
      return parsed.data;
    } catch (error) {
      console.error(`Error loading ${fileName}:`, error);
      return null;
    }
  };

  const analyzeData = () => {
    if (!data.multipleChoice || !data.schema) return;

    setLoading(true);
    
    try {
      // Clean headers by removing whitespace
      const cleanedData = data.multipleChoice.map(row => {
        const cleanedRow = {};
        Object.keys(row).forEach(key => {
          const cleanKey = key.trim();
          cleanedRow[cleanKey] = row[key];
        });
        return cleanedRow;
      });

      // 1. Demographics Analysis
      const genderDistribution = _.countBy(cleanedData, 'Q1');
      const ageDistribution = _.countBy(cleanedData, 'Q2');
      const educationDistribution = _.countBy(cleanedData, 'Q34');
      const roleDistribution = _.countBy(cleanedData, 'Q49');

      // 2. Programming Language Trends
      const primaryLanguages = _.countBy(cleanedData, 'Q17');
      const recommendedLanguages = _.countBy(cleanedData, 'Q18');

      // 3. Tool Usage Analysis
      const ides = _.countBy(cleanedData, 'Q13');
      const notebooks = _.countBy(cleanedData, 'Q14');
      const cloudServices = _.countBy(cleanedData, 'Q15');

      // 4. ML Framework Analysis
      const mlFrameworks = _.countBy(cleanedData, 'Q19');
      const mostUsedFramework = _.countBy(cleanedData, 'Q20');

      // 5. Experience Analysis
      const codingExperience = _.countBy(cleanedData, 'Q24');
      const mlExperience = _.countBy(cleanedData, 'Q25');

      // 6. Industry Analysis
      const industries = _.countBy(cleanedData, 'Q50');

      // Convert to chart-friendly format
      const formatForChart = (obj, limit = 10) => {
        return Object.entries(obj)
          .filter(([key]) => key && key !== 'undefined' && key !== '')
          .sort(([,a], [,b]) => b - a)
          .slice(0, limit)
          .map(([name, value]) => ({ name: name.length > 20 ? name.substring(0, 20) + '...' : name, value }));
      };

      const analysisResults = {
        demographics: {
          gender: formatForChart(genderDistribution),
          age: formatForChart(ageDistribution),
          education: formatForChart(educationDistribution),
          roles: formatForChart(roleDistribution)
        },
        programming: {
          primary: formatForChart(primaryLanguages),
          recommended: formatForChart(recommendedLanguages)
        },
        tools: {
          ides: formatForChart(ides),
          notebooks: formatForChart(notebooks),
          cloud: formatForChart(cloudServices)
        },
        ml: {
          frameworks: formatForChart(mlFrameworks),
          mostUsed: formatForChart(mostUsedFramework)
        },
        experience: {
          coding: formatForChart(codingExperience),
          ml: formatForChart(mlExperience)
        },
        industry: formatForChart(industries),
        totalResponses: cleanedData.length
      };

      setAnalysis(analysisResults);
    } catch (error) {
      console.error('Error analyzing data:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateForecast = (data, periods = 5) => {
    // Simple linear trend forecast (placeholder for more sophisticated models)
    if (!data || data.length < 3) return [];
    
    const values = data.map(d => d.value);
    const n = values.length;
    const sumX = (n * (n + 1)) / 2;
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = values.reduce((sum, y, i) => sum + y * (i + 1), 0);
    const sumXX = (n * (n + 1) * (2 * n + 1)) / 6;
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    const forecast = [];
    for (let i = 1; i <= periods; i++) {
      const futureValue = Math.max(0, intercept + slope * (n + i));
      forecast.push({
        name: `Year +${i}`,
        value: Math.round(futureValue),
        forecast: true
      });
    }
    
    return data.concat(forecast);
  };

  useEffect(() => {
    // Auto-load files if they exist
    const loadAllFiles = async () => {
      await Promise.all([
        loadFile('multipleChoiceResponses.csv', 'multipleChoice'),
        loadFile('SurveySchema.csv', 'schema'),
        loadFile('freeFormResponses.csv', 'freeForm')
      ]);
    };
    
    loadAllFiles();
  }, []);

  useEffect(() => {
    if (data.multipleChoice && data.schema) {
      analyzeData();
    }
  }, [data.multipleChoice, data.schema]);

  const renderChart = (data, type, title) => {
    if (!data || data.length === 0) return <div className="text-gray-500">No data available</div>;

    switch (type) {
      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        );
      
      case 'line':
        const forecastData = generateForecast(data);
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#8884d8" 
                strokeWidth={2}
                dot={{ fill: '#8884d8' }}
              />
            </LineChart>
          </ResponsiveContainer>
        );
      
      default:
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        );
    }
  };

  const TabButton = ({ id, icon: Icon, label, active, onClick }) => (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
        active ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      }`}
    >
      <Icon size={18} />
      <span>{label}</span>
    </button>
  );

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Kaggle Survey Trend Analysis & Forecasting
        </h1>
        <p className="text-gray-600 mb-4">
          Comprehensive analysis of machine learning and data science survey responses
        </p>
        
        {/* File Upload Status */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className={`p-3 rounded-lg border ${data.multipleChoice ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
            <div className="flex items-center space-x-2">
              <Upload size={16} />
              <span className="text-sm font-medium">Multiple Choice Responses</span>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {data.multipleChoice ? `${data.multipleChoice.length} responses loaded` : 'Not loaded'}
            </p>
          </div>
          <div className={`p-3 rounded-lg border ${data.schema ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
            <div className="flex items-center space-x-2">
              <Upload size={16} />
              <span className="text-sm font-medium">Survey Schema</span>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {data.schema ? 'Schema loaded' : 'Not loaded'}
            </p>
          </div>
          <div className={`p-3 rounded-lg border ${data.freeForm ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}>
            <div className="flex items-center space-x-2">
              <Upload size={16} />
              <span className="text-sm font-medium">Free Form Responses</span>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {data.freeForm ? `${data.freeForm.length} responses loaded` : 'Optional - Not loaded'}
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-2 mb-6 flex-wrap">
          <TabButton
            id="overview"
            icon={Users}
            label="Demographics"
            active={activeTab === 'overview'}
            onClick={() => setActiveTab('overview')}
          />
          <TabButton
            id="programming"
            icon={Code}
            label="Programming"
            active={activeTab === 'programming'}
            onClick={() => setActiveTab('programming')}
          />
          <TabButton
            id="tools"
            icon={BarChart3}
            label="Tools & Platforms"
            active={activeTab === 'tools'}
            onClick={() => setActiveTab('tools')}
          />
          <TabButton
            id="ml"
            icon={TrendingUp}
            label="ML Frameworks"
            active={activeTab === 'ml'}
            onClick={() => setActiveTab('ml')}
          />
          <TabButton
            id="industry"
            icon={Briefcase}
            label="Industry & Experience"
            active={activeTab === 'industry'}
            onClick={() => setActiveTab('industry')}
          />
          <TabButton
            id="forecast"
            icon={Calendar}
            label="Forecasting"
            active={activeTab === 'forecast'}
            onClick={() => setActiveTab('forecast')}
          />
        </div>
      </div>

      {loading && (
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Analyzing survey data...</p>
        </div>
      )}

      {analysis && !loading && (
        <div className="space-y-6">
          {/* Overview/Demographics Tab */}
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Gender Distribution</h3>
                {renderChart(analysis.demographics.gender, 'pie')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Age Distribution</h3>
                {renderChart(analysis.demographics.age, 'bar')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Education Level</h3>
                {renderChart(analysis.demographics.education, 'bar')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Job Roles</h3>
                {renderChart(analysis.demographics.roles, 'bar')}
              </div>
            </div>
          )}

          {/* Programming Tab */}
          {activeTab === 'programming' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Most Used Programming Languages</h3>
                {renderChart(analysis.programming.primary, 'bar')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Recommended Languages for Beginners</h3>
                {renderChart(analysis.programming.recommended, 'bar')}
              </div>
            </div>
          )}

          {/* Tools Tab */}
          {activeTab === 'tools' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">IDEs</h3>
                {renderChart(analysis.tools.ides, 'bar')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Notebooks</h3>
                {renderChart(analysis.tools.notebooks, 'bar')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Cloud Services</h3>
                {renderChart(analysis.tools.cloud, 'bar')}
              </div>
            </div>
          )}

          {/* ML Frameworks Tab */}
          {activeTab === 'ml' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">ML Frameworks Used</h3>
                {renderChart(analysis.ml.frameworks, 'bar')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Most Used ML Framework</h3>
                {renderChart(analysis.ml.mostUsed, 'bar')}
              </div>
            </div>
          )}

          {/* Industry Tab */}
          {activeTab === 'industry' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Industries</h3>
                {renderChart(analysis.industry, 'bar')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Coding Experience</h3>
                {renderChart(analysis.experience.coding, 'bar')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">ML Experience</h3>
                {renderChart(analysis.experience.ml, 'bar')}
              </div>
            </div>
          )}

          {/* Forecasting Tab */}
          {activeTab === 'forecast' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Programming Language Trend Forecast</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Projected trend based on historical usage patterns (5-year forecast)
                </p>
                {renderChart(analysis.programming.primary, 'line')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">ML Framework Adoption Forecast</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Predicted framework adoption trends
                </p>
                {renderChart(analysis.ml.frameworks, 'line')}
              </div>
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Forecasting Methodology</h3>
                <div className="text-sm text-gray-700 space-y-2">
                  <p><strong>Current Implementation:</strong> Simple linear trend analysis</p>
                  <p><strong>Recommended Enhancements:</strong></p>
                  <ul className="list-disc list-inside ml-4 space-y-1">
                    <li>ARIMA modeling for time-series analysis</li>
                    <li>Facebook Prophet for trend decomposition</li>
                    <li>LSTM neural networks for complex pattern recognition</li>
                    <li>Ensemble methods combining multiple forecasting approaches</li>
                  </ul>
                  <p><strong>Data Requirements:</strong> Multi-year survey data for robust forecasting</p>
                </div>
              </div>
            </div>
          )}

          {/* Summary Statistics */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Survey Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{analysis.totalResponses.toLocaleString()}</div>
                <div className="text-sm text-gray-600">Total Responses</div>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{analysis.programming.primary.length}</div>
                <div className="text-sm text-gray-600">Programming Languages</div>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{analysis.tools.cloud.length}</div>
                <div className="text-sm text-gray-600">Cloud Platforms</div>
              </div>
              <div className="p-4 bg-orange-50 rounded-lg">
                <div className="text-2xl font-bold text-orange-600">{analysis.industry.length}</div>
                <div className="text-sm text-gray-600">Industries</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {!analysis && !loading && (
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <Upload size={48} className="mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Waiting for Data</h3>
          <p className="text-gray-600">
            Upload your CSV files (multipleChoiceResponses.csv, SurveySchema.csv, freeFormResponses.csv) to begin analysis
          </p>
        </div>
      )}
    </div>
  );
};

export default KaggleSurveyAnalyzer;