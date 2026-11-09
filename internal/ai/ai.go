// Package ai provides AI service functionality for multiple providers
// Author: Done-0
// Created: 2025-08-31
package ai

import (
	"github.com/Done-0/gin-scaffold/configs"
	"github.com/Done-0/gin-scaffold/internal/ai/internal"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/provider"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/types"
)

type (
	AIManager          = internal.Manager
	ChatRequest        = types.ChatRequest
	ChatResponse       = types.ChatResponse
	ChatStreamResponse = types.ChatStreamResponse
	Choice             = types.Choice
	Message            = types.Message
	MessageDelta       = types.MessageDelta
	Provider           = provider.Provider
	StreamChoice       = types.StreamChoice
	Usage              = types.Usage
)

// New creates a new AI manager instance
func New(config *configs.Config) (*AIManager, error) {
	return internal.New(config)
}
